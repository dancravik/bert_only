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
import argparse
import itertools
import os
import sys

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.comet_utils import init_experiment
from common.data import LABELS_5CLASS, load_dataframes, TweetDataset, make_collate_fn
from common.engine import evaluate, train_epoch
from common.error_analysis import log_error_analysis
from common.metrics import compute_all_metrics, confusion_matrix_and_report
from common.utils import count_trainable_params, freeze_backbone, seed_everything


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


def run_training(cfg, model, train_loader, val_loader, lr, epochs, device, experiment):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=cfg["train"]["weight_decay"],
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg["train"]["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    best_eval = None
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        eval_out = evaluate(model, val_loader, device)
        m = compute_all_metrics(eval_out["labels"], eval_out["preds"], eval_out["proba"], LABELS_5CLASS)

        experiment.log_metric("train_loss", train_loss, epoch=epoch)
        experiment.log_metric("val_loss", eval_out["loss"], epoch=epoch)
        for k, v in m.items():
            experiment.log_metric(f"val_{k}", v, epoch=epoch)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={eval_out['loss']:.4f} "
              f"val_f1_macro={m['f1_macro']:.4f} val_acc={m['accuracy']:.4f}")

        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            best_eval = eval_out

    return best_f1, best_eval


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
        n_trainable, n_total = count_trainable_params(model)
        experiment.log_parameter("n_trainable_params", n_trainable)
        experiment.log_parameter("n_total_params", n_total)

        best_f1, _ = run_training(
            cfg, model, train_loader, val_loader, lr, cfg["search"]["search_epochs"], device, experiment
        )
        results.append({"learning_rate": lr, "batch_size": bs, "best_val_f1_macro": best_f1})
        experiment.log_metric("best_val_f1_macro", best_f1)
        experiment.end()

    results_df = pd.DataFrame(results).sort_values("best_val_f1_macro", ascending=False)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    results_df.to_csv(f"{cfg['output_dir']}/search_results.csv", index=False)
    print("\n=== search results (best first) ===")
    print(results_df.to_string(index=False))
    print("\nUpdate `final.learning_rate` / `final.batch_size` in config.yaml with the winner, "
          "then rerun with --mode final")


def final_run(cfg, tokenizer, train_df, val_df, test_df, device):
    model_tag = cfg["model"]["name"].replace("/", "_")
    experiment = init_experiment(cfg, f"exp4_{model_tag}_final")
    experiment.add_tag("final")

    lr = cfg["final"]["learning_rate"]
    bs = cfg["final"]["batch_size"]
    epochs = cfg["final"]["epochs"]

    model, train_loader, val_loader = build_model_and_loaders(cfg, bs, tokenizer, train_df, val_df, device)
    n_trainable, n_total = count_trainable_params(model)
    experiment.log_parameter("n_trainable_params", n_trainable)
    experiment.log_parameter("n_total_params", n_total)

    run_training(cfg, model, train_loader, val_loader, lr, epochs, device, experiment)

    # final evaluation on held-out test set
    collate = make_collate_fn(tokenizer)
    test_loader = DataLoader(
        TweetDataset(test_df, tokenizer, cfg["data"]["max_length"]),
        batch_size=bs, shuffle=False, collate_fn=collate, num_workers=cfg["train"]["num_workers"],
    )
    test_out = evaluate(model, test_loader, device)
    test_metrics = compute_all_metrics(test_out["labels"], test_out["preds"], test_out["proba"], LABELS_5CLASS)
    for k, v in test_metrics.items():
        experiment.log_metric(f"test_{k}", v)

    cm, report = confusion_matrix_and_report(test_out["labels"], test_out["preds"], LABELS_5CLASS)
    print("\n=== TEST classification report ===")
    print(report)
    experiment.log_confusion_matrix(matrix=cm.tolist(), labels=LABELS_5CLASS)
    experiment.log_text(report, metadata={"type": "classification_report"})

    os.makedirs(cfg["output_dir"], exist_ok=True)
    log_error_analysis(
        experiment, test_df, test_out, LABELS_5CLASS,
        cfg["error_analysis"]["top_k"], cfg["output_dir"], split_name="test",
    )

    model.save_pretrained(f"{cfg['output_dir']}/model")
    tokenizer.save_pretrained(f"{cfg['output_dir']}/model")
    experiment.end()


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
        final_run(cfg, tokenizer, train_df, val_df, test_df, device)


if __name__ == "__main__":
    main()
