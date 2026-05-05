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
    merged = merge_config(merged, {k: v for k, v in experiment_config.items() if k != 'defaults'})

    return merged


def set_by_dotted_path(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split('.')
    cur = config

    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]

    cur[keys[-1]] = value


def get_by_dotted_path(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur = config
    for key in dotted_key.split('.'):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def parse_cli_value(raw: str) -> Any:
    lower = raw.lower()

    if lower == 'true':
        return True
    if lower == 'false':
        return False
    if lower in {'none', 'null'}:
        return None

    try:
        return int(raw)
    except ValueError:
        pass

    try:
        return float(raw)
    except ValueError:
        pass

    return raw


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    config = deepcopy(config)

    if not overrides:
        return config

    for item in overrides:
        if '=' not in item:
            raise ValueError(
                f'invalid override format: {item}, '
                f'expected format: key=value, e.g. trainer.optimizer.lr=0.001'
            )

        key, raw_value = item.split("=", 1)
        value = parse_cli_value(raw_value)
        set_by_dotted_path(config, key, value)

    return config


__all__ = ['load_experiment_config', 'set_by_dotted_path', 'parse_cli_value', 'apply_overrides']
