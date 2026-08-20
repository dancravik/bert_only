# EXP 4 — apply the same pipeline (search -> final -> metrics -> Comet ->
# error analysis) to 4 other BERT-family architectures. One script, driven
# by whichever config you point it at:
#   python train.py --config configs/roberta.yaml    --mode search
#   python train.py --config configs/roberta.yaml    --mode final
#   python train.py --config configs/albert.yaml      --mode search
#   ... (albert.yaml / distilbert.yaml / electra.yaml)
# AutoModelForSequenceClassification / AutoTokenizer handle all four
# checkpoints transparently, so no per-model code branching is needed —
# only the config.yaml (and thus the Comet tags) differ per model.
import comet_ml  # noqa: F401  (before torch: comet auto-logging hook)
import argparse
import itertools
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.comet_utils import init_experiment
from common.data import load_dataframes, TweetDataset, make_collate_fn
from common.trainer import final_run, log_run_params, run_training
from common.utils import freeze_backbone, seed_everything


def build_model_and_loaders(cfg, batch_size, tokenizer, train_df, val_df, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"], num_labels=cfg["model"]["num_labels"]
    )
    if cfg["model"]["freeze_backbone"]:
        model = freeze_backbone(model)
    model.to(device)

    collate = make_collate_fn(tokenizer)
    train_loader = DataLoader(
        TweetDataset(train_df, tokenizer, cfg["data"]["max_length"]),
        batch_size=batch_size, shuffle=True, collate_fn=collate,
        num_workers=cfg["train"]["num_workers"],
    )
    val_loader = DataLoader(
        TweetDataset(val_df, tokenizer, cfg["data"]["max_length"]),
        batch_size=batch_size, shuffle=False, collate_fn=collate,
        num_workers=cfg["train"]["num_workers"],
    )
    return model, train_loader, val_loader


def save_and_print_results(results, output_dir):
    import pandas as pd
    results_df = pd.DataFrame(results).sort_values("best_val_f1_macro", ascending=False)
    path = f"{output_dir}/search_results.csv"
    results_df.to_csv(path, index=False)
    print("\n=== search results (best first) ===")
    print(results_df.to_string(index=False))
    print("\nUpdate `final.learning_rate` / `final.batch_size` in config.yaml with the winner, "
          f"then rerun with --mode final (results also in {path})")


def search(cfg, tokenizer, train_df, val_df, device):
    combos = list(itertools.product(cfg["search"]["learning_rates"], cfg["search"]["batch_sizes"]))
    results = []
    for lr, bs in combos:
        model_tag = cfg["model"]["name"].replace("/", "_")
        exp_name = f"exp4_{model_tag}_search_lr{lr}_bs{bs}"
        print(f"\n=== search trial: {exp_name} ===")
        experiment = init_experiment(cfg, exp_name)
        experiment.add_tag("search")
        model, train_loader, val_loader = build_model_and_loaders(cfg, bs, tokenizer, train_df, val_df, device)
        log_run_params(experiment, cfg, lr, bs, cfg["search"]["search_epochs"], model, mode="search")

        best_f1, best_epoch, _, _ = run_training(
            cfg, model, train_loader, val_loader, lr, cfg["search"]["search_epochs"], device, experiment
        )
        results.append({"learning_rate": lr, "batch_size": bs,
                        "best_val_f1_macro": best_f1, "best_epoch": best_epoch})
        experiment.log_metric("best_val_f1_macro", best_f1)
        experiment.end()

    os.makedirs(cfg["output_dir"], exist_ok=True)
    save_and_print_results(results, cfg["output_dir"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["search", "final"], required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    train_df, val_df, test_df = load_dataframes(
        cfg["data"]["train_csv"], cfg["data"]["test_csv"],
        cfg["data"]["val_size"], cfg["train"]["seed"],
    )
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    model_tag = cfg["model"]["name"].replace("/", "_")
    if args.mode == "search":
        search(cfg, tokenizer, train_df, val_df, device)
    else:
        experiment = init_experiment(cfg, f"exp4_{model_tag}_final")
        experiment.add_tag("final")
        bs = cfg["final"]["batch_size"]
        model, train_loader, val_loader = build_model_and_loaders(cfg, bs, tokenizer, train_df, val_df, device)
        log_run_params(experiment, cfg, cfg["final"]["learning_rate"], bs,
                       cfg["final"]["epochs"], model, mode="final")
        final_run(cfg, experiment, model, train_loader, val_loader, tokenizer, test_df, device)


if __name__ == "__main__":
    main()
