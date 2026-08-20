"""Shared training loop + evaluation/logging helpers.

Centralizes what used to be copy-pasted across exp1/exp2/exp3/exp4:
- run_training: trains N epochs, tracks best val macro-F1 and keeps a CPU
  copy of the weights of the best epoch. Final runs restore these weights
  before test evaluation, so late-epoch overfitting cannot hurt test metrics;
- evaluate_test_and_log: test metrics + confusion matrix + classification
  report + per-sample-loss error analysis + JSON metrics dump;
- log_run_params: logs lr/batch_size/epochs/model/params so every trial is
  directly comparable in the Comet experiment table.
"""
import json
import os

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .data import LABELS_5CLASS, TweetDataset, make_collate_fn
from .engine import evaluate, train_epoch
from .error_analysis import log_error_analysis
from .metrics import compute_all_metrics, confusion_matrix_and_report
from .utils import count_trainable_params


def log_run_params(experiment, cfg, lr, batch_size, epochs, model, mode):
    n_trainable, n_total = count_trainable_params(model)
    experiment.log_parameter("mode", mode)
    experiment.log_parameter("learning_rate", lr)
    experiment.log_parameter("batch_size", batch_size)
    experiment.log_parameter("epochs", epochs)
    experiment.log_parameter("model_name", cfg["model"]["name"])
    experiment.log_parameter("freeze_backbone", bool(cfg["model"].get("freeze_backbone", False)))
    experiment.log_parameter("n_trainable_params", n_trainable)
    experiment.log_parameter("n_total_params", n_total)
    print(f"[run] mode={mode} model={cfg['model']['name']} lr={lr} bs={batch_size} "
          f"epochs={epochs} trainable={n_trainable:,}/{n_total:,}")


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
    best_epoch = -1
    best_state = None
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        eval_out = evaluate(model, val_loader, device)
        m = compute_all_metrics(eval_out["labels"], eval_out["preds"], eval_out["proba"], LABELS_5CLASS)

        experiment.log_metric("train_loss", train_loss, epoch=epoch)
        experiment.log_metric("val_loss", eval_out["loss"], epoch=epoch)
        experiment.log_metric("lr", optimizer.param_groups[0]["lr"], epoch=epoch)
        for k, v in m.items():
            experiment.log_metric(f"val_{k}", v, epoch=epoch)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={eval_out['loss']:.4f} "
              f"val_f1_macro={m['f1_macro']:.4f} val_acc={m['accuracy']:.4f}")

        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            best_eval = eval_out
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return best_f1, best_epoch, best_state, best_eval


def final_run(cfg, experiment, model, train_loader, val_loader, tokenizer, test_df, device):
    """Full training + best-weights test evaluation + error analysis + model save."""
    lr = cfg["final"]["learning_rate"]
    bs = cfg["final"]["batch_size"]
    epochs = cfg["final"]["epochs"]

    best_f1, best_epoch, best_state, _ = run_training(
        cfg, model, train_loader, val_loader, lr, epochs, device, experiment
    )
    experiment.log_metric("best_val_f1_macro", best_f1)
    experiment.log_parameter("best_epoch", best_epoch)

    output_dir = cfg["output_dir"]
    evaluate_test_and_log(
        cfg, experiment, model, best_state, tokenizer, test_df, device, output_dir, bs
    )
    model.save_pretrained(f"{output_dir}/model")
    tokenizer.save_pretrained(f"{output_dir}/model")
    experiment.end()


def evaluate_test_and_log(cfg, experiment, model, state_dict, tokenizer, test_df, device,
                          output_dir, batch_size, split_name="test"):
    """Evaluate on the held-out test split with `state_dict` weights (usually
    the best-val epoch), log everything to Comet and dump files locally."""
    if state_dict is not None:
        model.load_state_dict(state_dict)

    collate = make_collate_fn(tokenizer)
    loader = DataLoader(
        TweetDataset(test_df, tokenizer, cfg["data"]["max_length"]),
        batch_size=batch_size, shuffle=False, collate_fn=collate,
        num_workers=cfg["train"]["num_workers"],
    )
    eval_out = evaluate(model, loader, device)
    metrics = compute_all_metrics(eval_out["labels"], eval_out["preds"], eval_out["proba"], LABELS_5CLASS)
    for k, v in metrics.items():
        experiment.log_metric(f"{split_name}_{k}", v)

    cm, report = confusion_matrix_and_report(eval_out["labels"], eval_out["preds"], LABELS_5CLASS)
    print(f"\n=== {split_name.upper()} classification report ===")
    print(report)
    experiment.log_confusion_matrix(matrix=cm.tolist(), labels=LABELS_5CLASS)
    experiment.log_text(report, metadata={"type": f"{split_name}_classification_report"})

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{split_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log_error_analysis(
        experiment, test_df, eval_out, LABELS_5CLASS,
        cfg["error_analysis"]["top_k"], output_dir, split_name=split_name,
    )
    return eval_out, metrics
