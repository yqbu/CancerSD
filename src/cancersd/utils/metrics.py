import numpy as np
import torch
from tqdm import tqdm

from scipy import stats
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, \
    precision_score, recall_score, f1_score, average_precision_score, euclidean_distances


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