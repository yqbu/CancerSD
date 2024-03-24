import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import utils
from auxiliary import PatientDataset, MetaTaskDataset
from model import MODALITY_meta


def main():
    omics_types = ['methylation', 'miRNA', 'mRNA']
    subtype_number_dict = {'CIN': 0, 'MSI': 1, 'GS': 2, 'EBV': 3}
    stad_dataset = PatientDataset(omics_types, cancer='STAD', subtype_number_dict=subtype_number_dict)
    acrg_dataset = PatientDataset(['mRNA'], cancer='ACRG', subtype_number_dict=subtype_number_dict,
                                  diag_drop_columns=['stage'])
    acrg_dataset = acrg_dataset.extend_as(stad_dataset)

    base_kwargs = {
        'omics_types': omics_types,
        'num_way': args.num_way,
        'num_shot': args.num_shot,
        'omics_dimensions': stad_dataset.omics_dimensions,
        'subtype_number_dict': subtype_number_dict
    }

    np.random.seed(0)
    seeds = np.random.randint(0, 65535, 10)
    evaluations = []
    accuracies = []
    test_acc = []
    start = time.time()
    for fold, seed in enumerate(seeds):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        print(f'============================== fold->{fold + 1} ==============================')
        train_dataset = stad_dataset
        ft_dataset, test_dataset = acrg_dataset.getNwayKshot(args.num_shot)

        # fine-tuning dataloader
        ft_dataloader = DataLoader(ft_dataset, args.num_way * args.num_shot)
        test_dataloader = DataLoader(test_dataset, len(test_dataset), shuffle=True)

        for ft_support_x, ft_support_y in ft_dataloader:
            ft_support_x = ft_support_x.squeeze(0).to(device)
            ft_support_y = ft_support_y.squeeze(0).to(device)

        for ft_query_x, ft_query_y in test_dataloader:
            ft_query_x = ft_query_x.squeeze(0).to(device)
            ft_query_y = ft_query_y.squeeze(0).to(device)

        meta = MODALITY_meta(args, train_dataset.omics_dimensions_dict).to(device)
        meta.clcLoss.to(device)
        print('meta train stage running...')
        for current_epoch in range(args.epoch):
            train_task_dataset = MetaTaskDataset(**base_kwargs, patients_info=train_dataset.patients_info)
            train_task_dataloader = DataLoader(train_task_dataset, args.num_task, shuffle=True, drop_last=True)

            print(f'----------------------------- epoch->{current_epoch + 1} ------------------------------')
            for support_x, support_y, query_x, query_y in train_task_dataloader:
                support_x = support_x.to(device)
                support_y = support_y.to(device)
                query_x = query_x.to(device)
                query_y = query_y.to(device)
                meta(support_x, support_y, query_x, query_y,
                     ft_support_x, ft_support_y, current_epoch,
                     (len(stad_dataset.omics_types) > len(acrg_dataset.omics_dimensions)))

        print('fine tune stage running...')
        acc, prediction, probability = meta.fine_tune(ft_support_x, ft_support_y, ft_query_x, ft_query_y,
                                                      args.ft_update_step)
        print(f'test acc: {acc}')
        test_acc.append(np.array(acc))
        print(f'test average acc: {np.array(acc).mean(axis=0)}')
        accuracies.append(acc[-1])
        prediction = prediction.tolist()
        probability = probability.cpu().numpy()
        evaluation = utils.get_performance_evaluation(ft_query_y.cpu(), prediction, probability)

        print('performance printing...')
        print(evaluation)
        for k, v in evaluation.items():
            print(f'{k}->{v}')
        evaluations.append(evaluation)
        print(f'spend time: {time.time() - start}')
        start = time.time()

        del meta

    print('accuracy:', utils.print_statistic(evaluations, 'accuracy'))
    print('auc:', utils.print_statistic(evaluations, 'auc'))
    print('recall:', utils.print_statistic(evaluations, 'recall'))
    print('precision:', utils.print_statistic(evaluations, 'precision'))
    print('F1 score:', utils.print_statistic(evaluations, 'F1 score'))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, default=60, help='')
    argparser.add_argument('--num_way', type=int, default=4, help='')
    argparser.add_argument('--num_shot', type=int, default=10, help='')
    argparser.add_argument('--num_task', type=int, default=4, help='')
    argparser.add_argument('--update_step', type=int, default=15, help='')
    argparser.add_argument('--ft_update_step', type=int, default=512, help='')
    argparser.add_argument('--weight_decay', type=float, default=2e-4, help='')
    argparser.add_argument('--inner_lr', type=float, default=2e-4, help='')
    argparser.add_argument('--outer_lr', type=float, default=2e-4, help='')
    argparser.add_argument('--fine_tuning_lr', type=float, default=2e-5, help='')
    argparser.add_argument('--embedding_dimension', type=int, default=512, help='')
    argparser.add_argument('--rank', type=int, default=8, help='')
    argparser.add_argument('--tau', type=float, default=0.5, help='')

    args = argparser.parse_args()
    print(f'start time->{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}')
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    main()
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"{key}: {value}")
