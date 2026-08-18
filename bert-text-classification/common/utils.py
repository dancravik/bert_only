import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_backbone(model, backbone_attr_candidates=("bert", "roberta", "albert", "distilbert", "electra", "deberta")):
    """Freezes every parameter except the classification head. Works across
    AutoModelForSequenceClassification variants by finding the encoder
    submodule by common attribute names, then freezing it; everything else
    (pooler if separate + classifier head) stays trainable."""
    backbone = None
    for attr in backbone_attr_candidates:
        if hasattr(model, attr):
            backbone = getattr(model, attr)
            break
    if backbone is None:
        raise ValueError(
            f"Could not find a known backbone attribute on {type(model)}; "
            f"checked {backbone_attr_candidates}. Inspect model.named_children() "
            "and add the right attribute name."
        )
    for param in backbone.parameters():
        param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[freeze_backbone] trainable params: {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.2f}%)")
    return model


def count_trainable_params(model):
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_trainable, n_total
