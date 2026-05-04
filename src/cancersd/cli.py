from __future__ import annotations

import argparse

from cancersd.engine.runner import *
from cancersd.infra.config import *


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run CancerSD experiments')

    parser.add_argument(
        '--config',
        type=str,
        default='experiment/stad_diagnosis.yaml',
        help='experiment config path relative to configs'
    )

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--log-root', type=str, default=None)
    parser.add_argument('--output-root', type=str, default=None)

    return parser


