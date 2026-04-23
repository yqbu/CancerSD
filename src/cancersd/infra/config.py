from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cancersd.utils.common import deepcopy
from cancersd.infra.paths import CONFIG_ROOT


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or dict()

    return data


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def load_experiment_config(experiment_path: str | Path) -> dict[str, Any]:
    experiment_path = Path(experiment_path)
    if not experiment_path.is_absolute():
        experiment_path = CONFIG_ROOT / experiment_path

    experiment_config = load_yaml(experiment_path)
    merged = {}

    for item in experiment_config.get('defaults', []):
        config_path = CONFIG_ROOT / item
        part_config = load_yaml(config_path)
        merged = merge_config(merged, part_config)

    # load the remaining non-default configurations
    merged = merge_config(merged, {k: v for k, v in experiment_config.items() if k != "defaults"})

    return merged


__all__ = ['load_experiment_config']
