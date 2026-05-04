from cancersd.engine.runner import run_experiment
from cancersd.infra.config import load_experiment_config


def main() -> None:
    cancer = 'stad'
    exp_cfg = load_experiment_config(f'experiments/{cancer}_diagnosis.yaml')
    run_experiment(exp_cfg)


if __name__ == "__main__":
    main()
