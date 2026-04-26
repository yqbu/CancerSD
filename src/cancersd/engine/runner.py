from __future__ import annotations

from typing import Any
import pprint

from cancersd.engine.trainer import Trainer
from cancersd.infra.paths import prepare_runtime_paths


def run_experiment(config: dict[str, Any]) -> None:
    runtime_paths = prepare_runtime_paths(config)

    print("=" * 80)
    print("Final config:")
    pprint.pprint(config)

    print("=" * 80)
    print("Runtime paths:")
    print(f"data_root    : {runtime_paths.data_root}")
    print(f"output_root  : {runtime_paths.output_root}")
    print(f"log_root     : {runtime_paths.log_root}")

    