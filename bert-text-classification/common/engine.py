import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = {k: v.to(device) for k, v in batch.items() if k != "idx"}
        optimizer.zero_grad()
        out = model(**batch)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * batch["labels"].size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns aggregate loss plus per-sample predictions/probabilities/losses
    so callers can both compute metrics and do error analysis without a
    second forward pass."""
    model.eval()
    all_idx, all_labels, all_preds, all_proba, all_losses = [], [], [], [], []
    total_loss = 0.0
    for batch in tqdm(loader, desc="eval", leave=False):
        idx = batch["idx"]
        model_inputs = {k: v.to(device) for k, v in batch.items() if k != "idx"}
        out = model(**model_inputs)
        logits = out.logits
        loss = out.loss
        total_loss += loss.item() * model_inputs["labels"].size(0)

        per_sample_loss = F.cross_entropy(logits, model_inputs["labels"], reduction="none")
        proba = F.softmax(logits, dim=-1)
        preds = proba.argmax(dim=-1)

        all_idx.extend(idx.tolist())
        all_labels.extend(model_inputs["labels"].cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_proba.extend(proba.cpu().tolist())
        all_losses.extend(per_sample_loss.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    return {
        "loss": avg_loss,
        "idx": all_idx,
        "labels": all_labels,
        "preds": all_preds,
        "proba": all_proba,
        "per_sample_loss": all_losses,
    }
