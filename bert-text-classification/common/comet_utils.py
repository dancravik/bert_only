"""Comet ML experiment setup. API key + workspace/project live in config.yaml
so nothing needs to be typed into the Kaggle notebook by hand.

Smoke testing without Comet: export COMET_ML_DISABLED=1 and train.py skips
all remote logging and returns a no-op logger with the same interface.
(Own env var instead of comet's built-in COMET_MODE so dummy-backend
auto-detection cannot produce a confusing partial session.)
"""
import os

import comet_ml  # keep this import above train.py's torch import


class _OfflineExperiment:
    """Minimal stand-in with the logged method surface used across train.py,
    so --mode search/final run unchanged without Comet connectivity."""

    def add_tag(self, *a, **k): pass
    def add_tags(self, *a, **k): pass
    def set_name(self, *a, **k): pass
    def log_parameter(self, *a, **k): pass
    def log_parameters(self, *a, **k): pass
    def log_metric(self, *a, **k): pass
    def log_metrics(self, *a, **k): pass
    def log_text(self, *a, **k): pass
    def log_table(self, *a, **k): pass
    def log_asset(self, *a, **k): pass
    def log_confusion_matrix(self, *a, **k): pass
    def end(self, *a, **k): pass


def comet_disabled() -> bool:
    return os.environ.get("COMET_ML_DISABLED", "") in ("1", "true", "yes", "on", "True")


def init_experiment(cfg: dict, experiment_name: str):
    if comet_disabled():
        print(f"[comet] COMET_ML_DISABLED set -> offline run (would be: {experiment_name})")
        return _OfflineExperiment()

    comet_cfg = cfg["comet"]
    experiment = comet_ml.Experiment(
        api_key=comet_cfg["api_key"],
        project_name=comet_cfg["project_name"],
        workspace=comet_cfg["workspace"],
        auto_metric_logging=False,   # we log metrics ourselves, explicitly
        auto_param_logging=False,
    )
    experiment.set_name(experiment_name)
    if comet_cfg.get("tags"):
        experiment.add_tags(comet_cfg["tags"])
    experiment.log_parameters(_flatten(cfg))
    return experiment


def _flatten(d: dict, parent_key: str = "") -> dict:
    items = {}
    for k, v in d.items():
        key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten(v, key))
        else:
            items[key] = v
    return items
