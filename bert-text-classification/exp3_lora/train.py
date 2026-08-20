# EXP 3 — parameter-efficient fine-tuning with LoRA (Hu et al., 2021,
# https://arxiv.org/abs/2106.09685): backbone frozen, small trainable low-rank
# adapters injected into the attention projections — a middle ground between
# exp1 (linear probe) and exp2 (full fine-tune).
#
# Kaggle gotcha: recent peft requires torchao>=0.16 while Kaggle images ship
# torchao 0.10 → ImportError inside get_peft_model. Fix in a notebook cell:
#     !pip uninstall -y torchao     (or: !pip install -q 'torchao>=0.16')
import comet_ml  # noqa: F401  (before torch: comet auto-logging hook)
import argparse
import itertools
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.comet_utils import init_experiment
from common.data import load_dataframes, TweetDataset, make_collate_fn
from common.trainer import final_run, log_run_params, run_training
from common.utils import seed_everything


def build_model_and_loaders(cfg, batch_size, tokenizer, train_df, val_df, device):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"], num_labels=cfg["model"]["num_labels"]
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        # train the (fresh, randomly initialized) classification head fully,
        # not as LoRA. Only "classifier": BertForSequenceClassification has no
        # top-level "pooler" attr (it lives at model.bert.pooler) and matching
        # it here behaved inconsistently across peft versions.
        modules_to_save=["classifier"],
    )
    try:
        model = get_peft_model(base_model, lora_cfg)
    except ImportError as e:
        if "torchao" in str(e):
            raise RuntimeError(
                "peft requires a newer torchao than this environment has "
                "(common on Kaggle: torchao 0.10 vs required >=0.16). "
                "Fix: run `!pip uninstall -y torchao` (or `!pip install -q 'torchao>=0.16'`) "
                "in a Kaggle notebook cell and rerun this script."
            ) from e
        raise
    model.print_trainable_parameters()
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
        exp_name = f"exp3_search_lr{lr}_bs{bs}"
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

    if args.mode == "search":
        search(cfg, tokenizer, train_df, val_df, device)
    else:
        experiment = init_experiment(cfg, "exp3_lora_final")
        experiment.add_tag("final")
        bs = cfg["final"]["batch_size"]
        model, train_loader, val_loader = build_model_and_loaders(cfg, bs, tokenizer, train_df, val_df, device)
        log_run_params(experiment, cfg, cfg["final"]["learning_rate"], bs,
                       cfg["final"]["epochs"], model, mode="final")
        final_run(cfg, experiment, model, train_loader, val_loader, tokenizer, test_df, device)


if __name__ == "__main__":
    main()
