from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional, Literal

import numpy as np
import pandas as pd
import torch

from cancersd.data.datasets.base import BaseDataset


class PatientDataset(BaseDataset):

    def __init__(
            self,
            omics_types: Sequence[str],
            *,
            dataset_root: Optional[str | Path] = None,
            diagnosis_file: Optional[str] = None,
            suffix: str = '',
            omics_state: Literal['both', 'complete', 'missing'] = 'both',
            race: Optional[str] = None,
            mode: str = 'include',
            diagnosis_drop_columns: Sequence[str] = ('stage', 'race'),
            label_column: str = 'subtype',
            patient_info: Optional[pd.DataFrame] = None,
            # omics_dimensions: Sequence[int] = None,
            # omics_dimensions_dict: dict[str, int] = None,
            subtype_number_dict: dict[str, int] = None,
            **_: object
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.diagnosis_file = diagnosis_file
        self.suffix = suffix
        self.omics_state = omics_state
        self.race = race
        self.mode = mode
        self.label_column = label_column
        self.subtype_number_dict = subtype_number_dict

        diagnosis_file = diagnosis_file or 'patient_diagnose.csv'
        if patient_info is None:
            if dataset_root is None:
                dataset_root = Path('data')
            patients_info, omics_dimensions, omics_dimensions_dict = self.load_data(
                root=dataset_root,
                omics_types=omics_types,
                diagnosis_file=diagnosis_file,
                suffix=suffix,
                omics_state=omics_state,
                race=race,
                mode=mode,
                diagnosis_drop_columns=diagnosis_drop_columns,
                label_column=label_column,
            )
            self._init_patient_table(
                omics_types=omics_types,
                patients_info=patients_info,
                omics_dimensions_dict=omics_dimensions_dict,
                omics_dimensions=omics_dimensions,
                subtype_number_dict=subtype_number_dict,
                label_column=label_column,
            )
            self.patients_info = self.patients_info.sort_index(axis=0)

    @staticmethod
    def _read_omics_csv(path: Path, omics_type: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f'omics file not found: {path}')
        frame = pd.read_csv(path)
        # as an insurance measure, this determination condition can be removed
        if omics_type not in frame.columns:
            raise ValueError(f'omics file {path} must contain an index column named "{omics_type}"')
        frame = frame.set_index(omics_type).T
        frame.index = frame.index.astype(str)
        frame = frame.sort_index(axis=1)

        return frame

    @classmethod
    def load_data(
            cls,
            *,
            root: Optional[str | Path],
            omics_types: Sequence[str],
            diagnosis_file: str,
            suffix: str = '',
            omics_state: Literal['both', 'complete', 'missing'] = 'both',
            race: Optional[str] = None,
            mode: str = 'include',
            diagnosis_drop_columns: Sequence[str] = ('stage', 'race'),
            label_column: str = 'subtype',
    ) -> tuple[pd.DataFrame, Sequence[int], dict[str, int]]:
        root = Path(root)
        suffix = suffix or ''
        omics_frames: list[pd.DataFrame] = []
        omics_dimensions: list[int] = []
        omics_dimensions_dict: dict[str, int] = {}
        patients_with_incomplete: set[str] = set()

        for omics_type in omics_types:
            missing_path = root / f'patient_lack_{omics_type}.npy'
            if missing_path.exists():
                missing_patients = np.load(missing_path, allow_pickle=True).tolist()
                patients_with_incomplete.update(map(str, missing_patients))

            omics_path = root / f'{omics_type}{suffix}.csv'
            omics_frame = cls._read_omics_csv(omics_path, omics_type)
            omics_frames.append(omics_frame)
            omics_dimensions.append(omics_frame.shape[1])
            omics_dimensions_dict[omics_type] = omics_frame.shape[1]

        omics_data = pd.concat(omics_frames, axis=1)

        diagnosis_path = root / diagnosis_file
        if not diagnosis_path.exists():
            raise FileNotFoundError(f'diagnosis file not found: {diagnosis_path}')
        diagnosis_frame = pd.read_csv(diagnosis_path)
        if 'aspect' not in diagnosis_frame.columns:
            raise ValueError(f'diagnosis file must contain an index column named "aspect"')
        diagnosis_frame = diagnosis_frame.set_index('aspect')

        common_patients = [patient for patient in omics_data.index.astype(str) if patient in diagnosis_frame.columns]
        if not common_patients:
            raise ValueError(f'no common patients between omics data and diagnosis file')
        omics_data = omics_data.loc[common_patients]
        diagnoses = diagnosis_frame.loc[:, common_patients].T

        if race is not None:
            if 'race' not in diagnoses.columns:
                raise ValueError(f'race filtering was requested, but "race" is absent in diagnosis file')
            if mode == 'include':
                diagnoses = diagnoses[diagnoses['race'] == race]
            elif mode == 'exclude':
                diagnoses = diagnoses[diagnoses['race'] != race]
            else:
                raise ValueError(f'mode must be either "include" or "exclude"')
            omics_data = omics_data.loc[diagnoses.index]

        diagnoses = diagnoses.drop(columns=list(diagnosis_drop_columns), errors='ignore')
        if label_column not in diagnoses.columns:
            raise ValueError(f'label column "{label_column}" is absent after dropping diagnostic columns')
        diagnoses = diagnoses.loc[:, label_column]

        if omics_state == 'complete':
            reserved_patients = [patient for patient in omics_data.index if patient not in patients_with_incomplete]
        elif omics_state == 'missing':
            reserved_patients = [patient for patient in omics_data.index if patient in patients_with_incomplete]
        else:
            reserved_patients = list(omics_data.index)

        omics_data = omics_data.loc[reserved_patients]
        diagnoses = diagnoses.loc[reserved_patients]
        patients_info = pd.concat([omics_data, diagnoses], axis=1).sort_index(axis=0)

        return patients_info, omics_dimensions, omics_dimensions_dict

    @classmethod
    def concat(
            cls,
            datasets: Sequence['PatientDataset'],
            *,
            subtype_number_dict: dict[str, int] = None,
    ) -> 'PatientDataset':
        if not datasets:
            raise ValueError('no datasets provided')
        first = datasets[0]
        for dataset in datasets[1:]:
            if dataset.omics_dimensions_dict != first.omics_dimensions_dict:
                raise ValueError('datasets have different dimensions')
        patients_info = pd.concat([dataset.patients_info for dataset in datasets], axis=0)

        return cls(
            omics_types=first.omics_types,
            patients_info=patients_info,
            omics_dimensions_dict=first.omics_dimensions_dict,
            subtype_number_dict=subtype_number_dict or first.subtype_number_dict,
            label_column=first.label_column
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.patients_info.iloc[index]
        x = row.loc[self.feature_columns].to_numpy(dtype=np.float32)
        y = self.subtype_number_dict[row.loc[self.label_column]]

        return {
            'features': torch.tensor(x, dtype=torch.float32),
            'label': torch.tensor(y, dtype=torch.long)
        }
