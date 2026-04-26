from __future__ import annotations

from itertools import accumulate
from collections import OrderedDict
from typing import Mapping, Sequence

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

    def init_patient_table(
            self,
            *,
            omics_types: Sequence[str],
            patient_info: pd.DataFrame,
            omics_dimensions_dict: dict[str, int] = None,
            omics_dimensions=None,
            subtype_number_dict=None,
            label_column: str = 'subtype'
    ) -> None:
        if patient_info is None or patient_info.empty:
            raise ValueError('patient_info must be not empty')
        if label_column not in patient_info.columns:
            raise ValueError(f'label_column {label_column} not in patient_info')

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

        feature_columns = [col for col in patient_info.columns if col != label_column]
        expected_feature_dim = sum(self.omics_dimensions)
        if len(feature_columns) != expected_feature_dim:
            raise ValueError(
                f'feature dimension mismatch, got {len(feature_columns)} feature columns, '
                f'expected {expected_feature_dim}'
            )
        self.patient_info = pd.concat(
            [patient_info.loc[:feature_columns], patient_info.loc[:, label_column]], axis=1
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
        return [col for col in self.patient_info.columns if col != self.label_column]

    @property
    def features_frame(self) -> pd.DataFrame:
        return self.patient_info.loc[:, self.feature_columns]

    @property
    def labels_series(self) -> pd.Series:
        return self.patient_info.loc[:, self.label_column]

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

    def __len__(self) -> int:
        return len(self.patients_info)
