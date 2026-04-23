import torch
import numpy as np
import pandas as pd


def mask(patient_info, omics_dimensions, missing_rate=0.3):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    num_samples = len(patient_info)
    num_omics = len(omics_dimensions)

    df_index = patient_info.index.values
    df_column = patient_info.columns.values[:-1]
    subtypes = patient_info.iloc[:, -1]

    masked_list = []
    missing_count = 0

    for i in range(num_samples):
        while True:
            not_mask_label = torch.rand((1, num_omics)).round()
            if missing_count / num_samples >= missing_rate:
                not_mask_label = torch.ones((1, num_omics))
            if not_mask_label.sum() >= 1:
                break
        if not_mask_label.sum() < num_omics:
            missing_count += 1

        masked_omics = [masked.repeat((1, omics_dimensions[i])) for i, masked in enumerate(not_mask_label[0])]
        masked_list.append(torch.cat(masked_omics, dim=1))

    masked_mat = torch.cat(masked_list, dim=0).numpy()

    patient_df = patient_info.fillna(0.0)
    patient_df.replace(0.0, -1, inplace=True)
    patient_mat = patient_df.iloc[:, :-1].to_numpy()

    masked_patient_mat = patient_mat * masked_mat
    masked_patient_df = pd.DataFrame(data=masked_patient_mat, index=df_index, columns=df_column)
    masked_patient_df.replace(0.0, np.NaN, inplace=True)
    masked_patient_df.replace(-1, 0.0, inplace=True)

    return pd.concat([masked_patient_df, subtypes], axis=1)


def get_omics_masking(features, omics_dimensions_dict):
    maskList = []
    numCol = len(omics_dimensions_dict)
    omics_types = list(omics_dimensions_dict.keys())

    n = features.shape[0]
    for i in range(n):
        while True:
            temp = torch.rand((1, numCol)).round()
            if temp.sum() >= 1:
                break
        mask = [masked.repeat((1, omics_dimensions_dict[omics_types[i]])) for i, masked in enumerate(temp[0])]
        maskList.append(torch.cat(mask, dim=1))
    if len(maskList) == 0:
        return features
    mask = torch.cat(maskList, dim=0).to(features.device)

    return mask * features


def get_molecular_mask(features, rate=0.9):
    prob = torch.full_like(features, rate)
    mask = torch.bernoulli(prob).to(features.device)

    return mask * features


def gaussian_blur(features, noise_rate=0.1):
    noisy = features + torch.randn_like(features) * noise_rate

    return noisy.to(features.device)


def mix_up(features, alpha=0.1):
    disorder = torch.randperm(features.shape[0])

    return (1-alpha) * features + alpha * features[disorder]


def scaling(features, low=0, high=2):
    scalingFactor = (high - low) * torch.rand(1, features.shape[1], device=features.device) + low
    scalingFactor = scalingFactor.repeat(features.shape[0], 1)

    return features * scalingFactor


def replacement(features, omics_dimensions_dict, labels):
    omics = torch.split(features, list(omics_dimensions_dict.values()), dim=1)
    subtypes = torch.unique(labels)

    results = []
    for subtype in subtypes:
        locations = torch.where(labels == subtype)
        temp_omics = [o[locations] for o in omics]
        disorder_omics = []
        for o in temp_omics:
            disorder = torch.randperm(len(locations))
            disorder_omics.append(o[disorder])

        results.append(torch.cat(disorder_omics, dim=1))

    return torch.cat(results, dim=0)
