import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as func
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from auxiliary import PatientDataset
from loss import ContrastiveLoss, BaseLoss
from model import MODALITY


def train(param_lr, data_loader, epoch):
    optimizer = torch.optim.AdamW(param_lr, weight_decay=model_kwargs['weight_decay'])
    schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    model.toggle_stage('train')
    for _ in tqdm(range(epoch)):
        for samples, labels in data_loader:
            samples = samples.to(device)
            labels = labels.to(device)

            complete, incomplete, available, origins, projections, reconstructed, generated, diagnoses = model(samples)

            generation_loss = torch.zeros(1).squeeze().to(device)
            if complete.numel() >= 1:
                complete = complete.view(-1)
                generation_loss += Loss_generation(reconstructed, origins[complete])
            if incomplete.numel() >= 1:
                incomplete = incomplete.view(-1)
                generation_loss += Loss_generation((generated * available)[incomplete], origins[incomplete])
            if complete.numel() + incomplete.numel() == 0:
                generation_loss += Loss_generation(reconstructed, origins)

            contrastive_loss = torch.zeros(1).squeeze().to(device)
            if complete.numel() >= 2:
                contrastive_loss = Loss_contrastive(projections)

            if complete.numel() + incomplete.numel() > 0:
                labels = torch.cat([labels[complete].repeat(2, ), labels], dim=0)
            diagnosis_loss = Loss_diagnosis(diagnoses, labels)

            base_loss = Loss_base(torch.cat([contrastive_loss.unsqueeze(0),
                                             generation_loss.unsqueeze(0),
                                             diagnosis_loss.unsqueeze(0)]))
            optimizer.zero_grad()
            base_loss.backward()
            optimizer.step()

        schedular.step()


if __name__ == '__main__':
    print(f'start time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}')
    cuda_use = torch.cuda.is_available()
    print(f'GPU: {cuda_use}')
    device = torch.device('cuda:0') if cuda_use else torch.device('cpu')

    learning_rate = 1e-4
    model_kwargs = {
        'weight_decay': 1e-3,
        'encoder_lr': learning_rate,
        'generator_lr': learning_rate,
        'classifier_lr': learning_rate,
        'rank': 60,
        'tau': 0.5,
        'total_epoch': 115,
        'embedding_dimension': 512
    }
    for key, value in model_kwargs.items():
        print(f'{key}: {value}')

    cancer = 'STAD'
    omics_types = ['methylation', 'miRNA', 'mRNA']
    subtype_number_dict = {'CIN': 0, 'MSI': 1, 'GS': 2, 'EBV': 3}
    folder_suffix = None
    dataset = PatientDataset(omics_types, cancer=cancer, subtype_number_dict=subtype_number_dict)

    omics_dimensions = dataset.omics_dimensions
    total_dim = np.array(omics_dimensions).sum()
    weight = torch.tensor(dataset.class_weight, device=device)

    np.random.seed(0)
    seeds = np.random.randint(0, 65535, 10)
    evaluations = []
    subtype_accuracy = []
    for opRound, seed in enumerate(seeds):
        print(f'=================================={opRound + 1}==================================')
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        training_ratio = 0.8
        train_dataset, test_dataset = dataset.random_split([training_ratio, 1 - training_ratio])

        model = MODALITY(dataset.num_subtype, dataset.omics_dimensions_dict, model_kwargs['embedding_dimension'],
                         model_kwargs['rank']).to(device)

        Loss_contrastive = ContrastiveLoss(model_kwargs['tau']).to(device)
        Loss_generation = nn.MSELoss()
        Loss_diagnosis = nn.CrossEntropyLoss(weight=weight)
        Loss_base = BaseLoss(coefficient=[1, 1, 1]).to(device)

        train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=4, pin_memory=True,
                                      drop_last=True)
        train([
            {'params': model.encoder.parameters(), 'lr': model_kwargs['encoder_lr']},
            {'params': model.projector.parameters(), 'lr': model_kwargs['encoder_lr']},
            {'params': model.generator.parameters(), 'lr': model_kwargs['generator_lr']},
            {'params': model.diagnostor.parameters(), 'lr': model_kwargs['classifier_lr']}
        ], train_dataloader, model_kwargs['total_epoch'])

        model.toggle_stage('test')
        test_dataloader = DataLoader(test_dataset, len(test_dataset), shuffle=True)

        corrects = [0.0] * dataset.num_subtype
        totals = [0.0] * dataset.num_subtype

        for patients, subtypes in test_dataloader:
            patients = patients.to(device)
            subtypes = subtypes.to(device)
            with torch.no_grad():
                similarity, diagnoses = model.test(patients)

                targets = subtypes.tolist()
                predictions = diagnoses.argmax(dim=-1).tolist()
                probabilities = func.softmax(diagnoses.detach(), dim=-1).cpu().numpy()

                for idx in range(len(targets)):
                    cur = targets[idx]
                    corrects[cur] += 1 if targets[idx] == predictions[idx] else 0
                    totals[cur] += 1

        corrects = np.array(corrects)
        totals = np.array(totals)

        subtype_accuracy.append(corrects / totals)

        evaluation = utils.get_performance_evaluation(targets, predictions, probabilities, 'multiple')
        evaluations.append(evaluation)
        print(evaluation)
        print(f'test average similarity: {similarity.mean():.3f}, std: {similarity.std():.3f}, '
              f'weight: {Loss_base.weight[0].detach().cpu().numpy()}')
        if folder_suffix is not None:
            torch.save(model.state_dict(), f'./model_params/model_params_{cancer}_{folder_suffix}{str(opRound)}.pkl')

    performance = {
        'accuracy': utils.get_statistic(evaluations, 'accuracy'),
        'auc': utils.get_statistic(evaluations, 'auc'),
        'recall': utils.get_statistic(evaluations, 'recall'),
        'precision': utils.get_statistic(evaluations, 'precision'),
        'f1_score': utils.get_statistic(evaluations, 'F1 score'),
    }

    print('accuracy:', performance['accuracy'])
    print('auc:', performance['auc'])
    print('recall:', performance['recall'])
    print('precision:', performance['precision'])
    print('F1 score:', performance['f1_score'])

    subtype_accuracy = np.array(subtype_accuracy)
    subtype_accuracy_mean = np.mean(subtype_accuracy, axis=0)
    subtype_accuracy_std = np.std(subtype_accuracy, axis=0)

    subtype_accuracy = list(zip(subtype_accuracy_mean, subtype_accuracy_std))
    print(f'----------Accuracy for different {cancer} subtypes-----------')
    for idx in range(len(subtype_accuracy)):
        print(f'{dataset.number_subtype_dict[idx]}: {subtype_accuracy[idx][0]} ± {subtype_accuracy[idx][1]}')

    utils.format_model_info(model.print_model(), model_kwargs, performance)
