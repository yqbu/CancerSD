from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from cancersd.data.datasets.standard import PatientDataset


def build_dataloaders(config: dict[str, Any], runtime_paths) -> dict[str, DataLoader]:
    dataset_cfg = config['dataset']
    trainer_cfg = config['experiment']['trainer']
    runtime_cfg = config['runtime']

    dataset_dir = runtime_paths.data_root / dataset_cfg.get('root')
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    dataset = PatientDataset(
        omics_types=list(dataset_cfg['omics']),
        dataset_root=dataset_dir,
        diagnosis_file=dataset_cfg['diagnosis_file'],
        suffix=dataset_cfg['suffix'],
        omics_state=dataset_cfg['omics_state'],
        race=dataset_cfg['race'],
        mode=dataset_cfg['mode'],
        label_column=dataset_cfg['label_column'],
        subtype_number_dict=dataset_cfg['subtype_map'],
    )
    split_cfg = dataset_cfg.get("split", {})
    split_ratios = [split_cfg['train_ratio'], split_cfg['validation_ratio'], split_cfg['test_ratio']]
    datasets = dataset.stratified_split(
        ratios=split_ratios, seed=split_cfg['seed'], shuffle_result=dataset_cfg['shuffle_result']
    )

    batch_size = trainer_cfg['batch_size']

    train_loader = DataLoader(
        dataset=datasets[0],
        batch_size=batch_size,
        shuffle=True,
        num_workers=runtime_cfg['num_workers'],
        pin_memory=runtime_cfg['pin_memory'],
        drop_last=runtime_cfg['drop_last']
    )

    test_loader = DataLoader(
        dataset=datasets[-1],
        batch_size=batch_size,
        shuffle=False,
        num_workers=runtime_cfg['num_workers'],
        pin_memory=runtime_cfg['pin_memory']
    )

    base_info = {
        'subtype_count': dataset.num_subtype,
        'omics_dimensions_dict': dataset.omics_dimensions_dict
    }
    if len(datasets) == 3:
        validation_loader = DataLoader(
            dataset=datasets[1],
            batch_size=batch_size,
            shuffle=True,
            num_workers=runtime_cfg['num_workers'],
            pin_memory=runtime_cfg['pin_memory'],
            drop_last=runtime_cfg['drop_last']
        )
        return {
            **base_info,
            'train': train_loader,
            'validation': validation_loader,
            'test': test_loader
        }
    else:
        return {
            **base_info,
            'train': train_loader,
            'test': test_loader
        }
