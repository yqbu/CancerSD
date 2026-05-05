from __future__ import annotations

import argparse

from cancersd.engine.runner import *
from cancersd.infra.config import *


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run CancerSD experiments')

    parser.add_argument('--cancer', type=str, default='STAD')
    # parser.add_argument(
    #     '--config',
    #     type=str,
    #     default='experiments/stad_diagnosis.yaml',
    #     help='experiment config path relative to configs'
    # )

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--log-root', type=str, default=None)
    parser.add_argument('--output-root', type=str, default=None)

    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--missing-rate', type=float, default=None)

    parser.add_argument(
        '--set',
        action='append',
        default=[],
        help='override config item, e.g. --set trainer.optimizer.lr=0.001',
    )

    return parser


def apply_named_args(config: dict, args: argparse.Namespace) -> dict:
    """
    override the commonly used `argparse` parameters into `cfg`
    only override the YAML when the parameters are not `None`

    :param config:
    :param args:
    :return:
    """
    mapping = {
        'seed': 'seed',
        'device': 'device',
        'data_root': 'paths.data_root',
        'output_root': 'paths.output_root',
        'log_root': 'paths.log_root',
        'run_name': 'experiment.name',
        'lr': 'experiment.trainer.optimizer.lr',
        'batch_size': 'experiment.trainer.batch_size',
        'epochs': 'experiment.trainer.epochs',
        'missing_rate': 'dataset.missing.missing_rate',
    }

    for arg_name, cfg_key in mapping.items():
        value = getattr(args, arg_name)
        if value is not None:
            set_by_dotted_path(config, cfg_key, value)

    return config


def main():
    parser = build_parser()
    args = parser.parse_args()

    cancer = args.cancer.lower()
    config = load_experiment_config(f'experiments/{cancer}_diagnosis.yaml')

    # common command-line parameter overwriting
    config = apply_named_args(config, args)
    # --set override
    config = apply_overrides(config, args.set)

    run_experiment(config)


if __name__ == "__main__":
    main()
