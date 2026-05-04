from __future__ import annotations

from itertools import accumulate
from collections import OrderedDict
from typing import Mapping, Sequence, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def cumulative_sum(nums: Sequence[int]) -> list[int]:
    """
    return cumulative boundaries, e.g. [2, 3] -> [0, 2, 5]
    """
    return [0, *accumulate(nums)]


class BaseDataset(Dataset):

    def _init_patient_table(
            self,
            *,
            omics_types: Sequence[str],
            patients_info: pd.DataFrame,
            omics_dimensions_dict: dict[str, int] = None,
            omics_dimensions=None,
            subtype_number_dict=None,
            label_column: str = 'subtype'
    ) -> None:
        if patients_info is None or patients_info.empty:
            raise ValueError('patients_info must be not empty')
        if label_column not in patients_info.columns:
            raise ValueError(f'label_column {label_column} not in patients_info')

        self.omics_types = list(omics_types)
        self.label_column = label_column

        if omics_dimensions_dict is None:
            if omics_dimensions is None:
                raise ValueError('omics_dimensions_dict or omics_dimensions must be provided')
            if len(omics_dimensions) != len(self.omics_types):
                raise ValueError('omics_dimensions must have same length as omics_types')
            omics_dimensions_dict = dict(zip(self.omics_types, map(int, omics_dimensions)))
        self.omics_dimensions_dict = OrderedDict(
            (omics_type, int(omics_dimensions_dict[omics_type]))
            for omics_type in self.omics_types
        )
        self.omics_dimensions = list(self.omics_dimensions_dict.values())

        feature_columns = [col for col in patients_info.columns if col != label_column]
        expected_feature_dim = sum(self.omics_dimensions)
        if len(feature_columns) != expected_feature_dim:
            raise ValueError(
                f'feature dimension mismatch, got {len(feature_columns)} feature columns, '
                f'expected {expected_feature_dim}'
            )
        # print(patients_info)
        self.patients_info = pd.concat(
            [patients_info.loc[:, feature_columns], patients_info.loc[:, [label_column]]], axis=1
        ).copy()

        if subtype_number_dict is None:
            subtype_names = list(dict.fromkeys(self.patients_info[label_column].tolist()))
            subtype_number_dict = {subtype: idx for idx, subtype in enumerate(subtype_names)}

        self.subtype_number_dict = dict(subtype_number_dict)
        self.number_subtype_dict = {v: k for k, v in self.subtype_number_dict.items()}
        self.num_subtype = len(self.subtype_number_dict)

        counts = self.class_counts
        num_samples = len(self.patients_info)
        num_classes = len(self.subtype_number_dict)
        weights = [0.0] * num_classes
        for subtype, count in counts.items():
            if subtype not in self.subtype_number_dict:
                continue
            weights[self.subtype_number_dict[subtype]] = num_samples / (count * num_classes)
        self.class_weights = torch.Tensor(weights)

    @property
    def feature_columns(self) -> list[str]:
        return [col for col in self.patients_info.columns if col != self.label_column]

    @property
    def features_frame(self) -> pd.DataFrame:
        return self.patients_info.loc[:, self.feature_columns]

    @property
    def labels_series(self) -> pd.Series:
        return self.patients_info.loc[:, self.label_column]

    @property
    def class_counts(self) -> dict[str, int]:
        return self.labels_series.value_counts().to_dict()

    def omics_slices(self) -> dict[str, slice]:
        """
        fragment-wise index mapping of multi-omics data, e.g.:
        {
            'mRNA':            slice(0, 100),
            'miRNA':           slice(100, 150),
            'methylation': slice(150, 350)
        }
        :return:
        """
        boundaries = cumulative_sum(self.omics_dimensions)
        return {
            omics_type: slice(boundaries[i], boundaries[i + 1])
            for i, omics_type in enumerate(self.omics_types)
        }

    def omics_frame(self, omics_type: str) -> pd.DataFrame:
        if omics_type not in self.omics_dimensions_dict:
            raise ValueError(f'unknown omics type: {omics_type}')

        return self.features_frame.iloc[:, self.omics_slices()[omics_type]]

    def _spawn(
            self,
            patients_info: pd.DataFrame,
            *,
            omics_types: Sequence[str] | None = None,
            omics_dimensions: Sequence[int] | None = None,
            omics_dimensions_dict: dict[str, int] | None = None,
            subtype_number_dict: dict[str, int] | None = None,
    ):
        return self.__class__(
            omics_types=list(omics_types or self.omics_types),
            dataset_root=self.dataset_root,
            diagnosis_file=self.diagnosis_file,
            suffix=self.suffix,
            omics_state=self.omics_state,
            race=self.race,
            mode=self.mode,
            patients_info=patients_info.copy(),
            omics_dimensions=list(omics_dimensions or self.omics_dimensions),
            omics_dimensions_dict=dict(omics_dimensions_dict or self.omics_dimensions_dict),
            subtype_number_dict=dict(subtype_number_dict or self.subtype_number_dict),
            label_column=self.label_column
        )

    def stratified_split(self, ratios: Sequence[float], *, seed: Optional[int] = None, shuffle_result: bool = True):
        ratios = np.asarray(ratios, dtype=float)
        if ratios.ndim != 1 or len(ratios) < 2:
            raise ValueError('ratios must contain at least 2 values')
        if np.any(ratios < 0):
            raise ValueError('ratios must be non-negative')
        ratios = ratios / ratios.sum()

        rng = np.random.default_rng(seed)
        split_frames = [[] for _ in ratios]
        for subtype in self.subtype_number_dict.keys():
            subtype_df = self.patients_info[self.labels_series == subtype]
            if subtype_df.empty:
                continue
            positions = np.arange(len(subtype_df))
            rng.shuffle(positions)
            cut_points = np.round(np.cumsum(ratios[:-1]) * len(subtype_df)).astype(int)
            for split_idx, pos_idx in enumerate(np.split(positions, cut_points)):
                split_frames[split_idx].append(subtype_df.iloc[pos_idx])

        outputs = []
        for frames in split_frames:
            split_df = pd.concat(frames, axis=0) if frames else self.patients_info.iloc[[]].copy()
            if split_df.empty:
                continue
            if shuffle_result and len(split_df) > 0:
                split_df = split_df.sample(frac=1.0, random_state=seed)
            outputs.append(self._spawn(split_df))
        return tuple(outputs)

    def random_split(self, ratios: Sequence[float], seed: Optional[int] = None):
        if len(ratios) != 2:
            raise ValueError('ratios must contain exactly 2 values, e.g. [0.8, 0.2]')
        return self.stratified_split(ratios, seed=seed)

    def train_validation_test(self, ratios: Sequence[float], seed: int | None = None):
        if len(ratios) != 3:
            raise ValueError('ratios must contain exactly 2 values, e.g. [0.7, 0.1, 0.2]')
        return self.stratified_split(ratios, seed=seed)

    def get_batch(self, batch_sz: int, device: Optional[torch.device | str] = None, seed: Optional[int] = None):
        if batch_sz <= 0:
            raise ValueError('batch size must be positive')

        rng = np.random.default_rng(seed)
        positions = rng.choice(len(self.patients_info), size=min(batch_sz, len(self.patients_info)), replace=False)

        x = torch.tensor(self.features_frame.iloc[positions].to_numpy(dtype=np.float32), dtype=torch.float)
        y = torch.tensor([self.subtype_number_dict[label] for label in self.labels_series.iloc[positions]], dtype=torch.long)

        if device is not None:
            x = x.to(device=device)
            y = y.to(device=device)

        return x, y

    def __len__(self) -> int:
        return len(self.patients_info)
