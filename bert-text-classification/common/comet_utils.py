"""Comet ML experiment setup. API key + workspace/project live in config.yaml
so nothing needs to be typed into the Kaggle notebook by hand."""
import comet_ml  # must be imported before torch for auto-logging hooks


def init_experiment(cfg: dict, experiment_name: str):
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
