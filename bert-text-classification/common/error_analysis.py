import pandas as pd


def top_loss_table(df_source: pd.DataFrame, eval_out: dict, class_names, top_k: int = 25):
    """df_source must be the exact dataframe the eval Dataset was built from
    (row order == idx used during tokenization)."""
    rows = []
    for idx, label, pred, loss, proba in zip(
        eval_out["idx"], eval_out["labels"], eval_out["preds"],
        eval_out["per_sample_loss"], eval_out["proba"]
    ):
        rows.append({
            "text": df_source.iloc[idx]["text"],
            "true_label": class_names[label],
            "pred_label": class_names[pred],
            "loss": loss,
            "confidence_in_pred": proba[pred],
            "correct": label == pred,
        })
    out = pd.DataFrame(rows)
    hardest = out.sort_values("loss", ascending=False).head(top_k).reset_index(drop=True)
    easiest = out.sort_values("loss", ascending=True).head(top_k).reset_index(drop=True)
    return out, hardest, easiest


def log_error_analysis(experiment, df_source, eval_out, class_names, top_k, out_dir, split_name="val"):
    full, hardest, easiest = top_loss_table(df_source, eval_out, class_names, top_k)

    full_path = f"{out_dir}/{split_name}_all_losses.csv"
    hard_path = f"{out_dir}/{split_name}_top{top_k}_hardest.csv"
    easy_path = f"{out_dir}/{split_name}_top{top_k}_easiest.csv"
    full.to_csv(full_path, index=False)
    hardest.to_csv(hard_path, index=False)
    easiest.to_csv(easy_path, index=False)

    experiment.log_table(f"{split_name}_hardest_examples.csv", hardest)
    experiment.log_table(f"{split_name}_easiest_examples.csv", easiest)
    experiment.log_asset(full_path)

    print(f"\n=== {split_name}: {top_k} HARDEST examples (highest loss) ===")
    print(hardest[["true_label", "pred_label", "loss", "text"]].to_string(index=False))
    print(f"\n=== {split_name}: {top_k} EASIEST examples (lowest loss) ===")
    print(easiest[["true_label", "pred_label", "loss", "text"]].to_string(index=False))

    return full, hardest, easiest
