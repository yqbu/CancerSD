import os
import tabnanny
import sys
import time

from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy import stats
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, \
    precision_score, recall_score, f1_score, average_precision_score, euclidean_distances


rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']


def t_test(sam1, sam2, independence=True):
    homoscedasticity = stats.levene(sam1, sam2)
    if independence:
        r = stats.ttest_ind(sam1, sam2, equal_var=(homoscedasticity[1] > 0.05))
    else:
        r = stats.ttest_rel(sam1, sam2)

    statistic = r.__getattribute__("statistic")
    pvalue = r.__getattribute__("pvalue")

    better = 0 if (statistic > 0) else 1

    return better, pvalue


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


def cumulative_sum(nums):
    current_sum = 0
    results = [current_sum]
    for i in range(len(nums)):
        current_sum += nums[i]
        results.append(current_sum)

    return results


def get_confusion_mat(targets, predictions):
    targets = torch.cat(targets, dim=-1)
    predictions = torch.cat(predictions, dim=-1)
    return confusion_matrix(targets, predictions)


def get_performance_evaluation(targets, predictions, probabilities, mode='multiple'):
    if mode == 'multiple':
        evaluation = {
            'accuracy': accuracy_score(targets, predictions),
            'auc': roc_auc_score(targets, probabilities, multi_class='ovo', average='weighted'),
            'precision': precision_score(targets, predictions, average='weighted', zero_division=0),
            'recall': recall_score(targets, predictions, average='weighted'),
            'F1 score': f1_score(targets, predictions, average='weighted')
        }
    elif mode == 'binary':
        probabilities = [probability[1] for probability in probabilities]
        evaluation = {
            'accuracy': accuracy_score(targets, predictions),
            'auc': roc_auc_score(targets, probabilities),
            'auprc': average_precision_score(targets, probabilities),
            'precision': precision_score(targets, predictions, zero_division=0),
            'recall': recall_score(targets, predictions),
            'F1 score': f1_score(targets, predictions)
        }

    return evaluation


def get_statistic(evaluations, measure):
    data = np.asarray([evaluation[measure] for evaluation in evaluations])

    return data.mean(), data.std()


def print_statistic(evaluations, measure):
    data = np.asarray([evaluation[measure] for evaluation in evaluations])

    return f'mean: {data.mean()}, std: {data.std()}'


def get_similarity(dataset):
    patient_count = dataset.__len__()
    np.set_printoptions(threshold=patient_count * patient_count)
    similarity = np.zeros((patient_count, patient_count))

    for i in tqdm(range(patient_count)):
        pi, _ = dataset[i]
        for j in range(patient_count):
            pj, _ = dataset[j]
            similarity[i][j] = torch.cosine_similarity(pi, pj, dim=-1)

    return similarity


def format_model_info(model_dict, hyper_parameters=None, performance=None, log_file='screenshot.log'):
    temp = sys.stdout
    log = open(log_file, 'a')
    sys.stdout = log

    headers = ['name', 'layer', 'parameters']
    current = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    print(f'model structure, start time->{current}')
    print(f'{headers[0]:<32}{headers[1]:<12}{headers[2]:<20}')

    for k, config in model_dict.items():
        for index, params in enumerate(config):
            layer, param = params[0], str(params[1])
            if index == 0:
                print(f'{k:<32}{layer:<12}{param:<20}')
            else:
                print(f'{" ":<32}{layer:<12}{param:<20}')

    if hyper_parameters is not None:
        for k, v in hyper_parameters.items():
            print(f'{k} -> {v}')

    if performance is not None:
        for k, v in performance.items():
            print(f'{k} -> {v}')

    print()

    sys.stdout = temp
    log.close()


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
