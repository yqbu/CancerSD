from __future__ import annotations

from typing import Any
import pprint

from cancersd.data.dataloaders import build_dataloaders
from cancersd.engine.trainer import Trainer
from cancersd.infra.paths import prepare_runtime_paths, build_experiment_paths
from cancersd.models.model import CancerSD


def run_experiment(config: dict[str, Any]) -> None:
    runtime_paths = prepare_runtime_paths(config)
    experiment_paths = build_experiment_paths(config, runtime_paths)

    print("=" * 80)
    print("Final config:")
    pprint.pprint(config)

    print("=" * 80)
    print("Runtime paths:")
    print(f"data_root    : {runtime_paths.data_root}")
    print(f"output_root  : {runtime_paths.output_root}")
    print(f"log_root     : {runtime_paths.log_root}")

    train_cfg = config['experiment']['trainer']

    dataloaders = build_dataloaders(config, runtime_paths)
    model = CancerSD(
        num_way=dataloaders['subtype_count'],
        omics_dimensions_dict=dataloaders['omics_dimensions_dict'],
        embedding_dimension=train_cfg['hyper_parameters']['embedding_dimension'],
        rank=train_cfg['hyper_parameters']['rank']
    )

    task = train_cfg['task']
    if task == 'diagnosis':
        trainer = Trainer(config, model, dataloaders, experiment_paths)
    elif task == 'meta':
        pass
    else:
        raise ValueError(f'unknown task: {task}')

    trainer.fit()
    trainer.test()