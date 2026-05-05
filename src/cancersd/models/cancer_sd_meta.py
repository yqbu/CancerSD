import torch
from torch import nn
from torch.nn import functional as func
from tqdm import tqdm
import numpy as np

from cancersd.models.model import CancerSD
from cancersd.losses.loss import ContrastiveLoss, CategoryLevelContrastive
from cancersd.utils.common import deepcopy

class CancerSD_meta(nn.Module):

    def __init__(self, args, omics_dimensions_dict):
        super(CancerSD_meta, self).__init__()
        self.weight_decay = args.weight_decay
        self.inner_lr = args.inner_lr
        self.outer_lr = args.outer_lr
        self.fine_tuning_lr = args.fine_tuning_lr
        self.num_way = args.num_way
        self.num_shot = args.num_shot
        self.update_step = args.update_step
        self.ft_update_step = args.ft_update_step
        self.inner_epoch = args.epoch
        self.embedding_dimension = args.embedding_dimension
        self.omics_dimensions_dict = omics_dimensions_dict
        self.total_dimension = np.array(list(omics_dimensions_dict.values())).sum()
        self.rank = args.rank

        self.learner = CancerSD(self.num_way, self.omics_dimensions_dict, self.embedding_dimension, self.rank)

        self.ilcLoss = ContrastiveLoss(args.tau)
        self.loss_diagnosis = nn.CrossEntropyLoss()
        self.clcLoss = CategoryLevelContrastive()

        self.meta_optimizer = torch.optim.AdamW(self.learner.parameters(), lr=self.outer_lr,
                                                weight_decay=self.weight_decay)
        self.meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.meta_optimizer, T_0=1, T_mult=2)

    def get_prototype(self, x, y):
        embeddings = x.squeeze()
        embeddings_norm = embeddings / torch.norm(embeddings, p=2, dim=1).unsqueeze(1)
        em_sort = torch.sort(y)
        embeddings_norm = embeddings_norm[em_sort.indices]

        num_one = torch.sum(y)
        num_zero = y.shape[0] - num_one

        em_zero = embeddings_norm[:num_zero]
        em_one = embeddings_norm[num_zero:]

        proto_zero = em_zero.mean(dim=0).unsqueeze(0)
        proto_one = em_one.mean(dim=0).unsqueeze(0)
        proto_embeddings = torch.cat([proto_zero, proto_one], dim=0)

        return proto_embeddings

    def similarity_clc(self, input1, y1, input2, y2):
        proto_A = self.get_prototype(input1, y1)
        proto_B = self.get_prototype(input2, y2)

        return self.ilcLoss([proto_A, proto_B])

    def cal_loss(self, results, patients, subtypes, fine_tune_x=None, fine_tune_y=None, clc=False):
        complete, incomplete, available, origins, projections, reconstructed, generated, diagnoses = results

        recon_loss = 0
        if len(complete) > 0:
            recon_loss += func.mse_loss(reconstructed, origins[complete])
        if len(incomplete) > 0:
            recon_loss += func.mse_loss((generated * available)[incomplete], origins[incomplete])
        generation_loss = recon_loss

        if len(complete) >= 2:
            contrastive_loss = self.ilcLoss(projections)
        else:
            contrastive_loss = torch.zeros(1).to(patients.device)

        multi_labels = deepcopy(subtypes)
        if complete.numel() + incomplete.numel() > 0:
            multi_labels = torch.cat([subtypes[complete], subtypes[complete], subtypes], dim=0)
        diagnosis_loss = self.loss_diagnosis(diagnoses, multi_labels)

        total_loss = contrastive_loss + generation_loss + diagnosis_loss

        if clc and fine_tune_x is not None:
            embeddings1 = self.learner.encode(patients)
            embeddings2 = self.learner.encode(fine_tune_x)

            # distribution-based category-level contrastive loss
            total_loss += self.clcLoss(embeddings1, subtypes, embeddings2, fine_tune_y)

            # # similarity-based category-level contrastive loss
            # total_loss += self.similarity_clc(embeddings1, subtypes, embeddings2, fine_tune_y)

        return total_loss

    def forward(self, support_x, support_y, query_x, query_y, ft_x, ft_y, epoch, multi_label=True):
        num_task, batch_size, fea_dim = support_x.size()

        losses = [0] * (self.update_step + 1)
        corrects = [0] * (self.update_step + 1)
        totals = [0] * (self.update_step + 1)

        parameters = self.learner.parameters()

        clc_flag = True
        second_derivative_flag = (epoch > self.inner_epoch // 2)

        for i in range(num_task):
            lr = self.inner_lr
            results = self.learner(support_x[i], params=None)
            loss = self.cal_loss(results, support_x[i], support_y[i])

            grads = torch.autograd.grad(loss, parameters, retain_graph=second_derivative_flag,
                                        create_graph=second_derivative_flag, allow_unused=True)
            theta = list(map(lambda p: p[1] - lr * p[0] if p[0] is not None else p[1], zip(grads, parameters)))

            with torch.no_grad():
                results = self.learner(query_x[i], params=parameters)
                loss_before_update = self.cal_loss(results, query_x[i], query_y[i], ft_x, ft_y, clc=clc_flag)
                losses[0] += loss_before_update

                temp_labels = torch.cat([query_y[i][results[0]], query_y[i][results[0]], query_y[i]],
                                        dim=0) if multi_label else query_y[i]
                correct = torch.eq(results[-1].argmax(dim=1), temp_labels).sum().item()
                corrects[0] += round(correct, 4)
                totals[0] += temp_labels.shape[0]

            results = self.learner(query_x[i], params=theta)
            loss_after_update = self.cal_loss(results, query_x[i], query_y[i], ft_x, ft_y, clc=clc_flag)
            losses[1] += loss_after_update

            temp_labels = torch.cat([query_y[i][results[0]], query_y[i][results[0]], query_y[i]],
                                    dim=0) if multi_label else query_y[i]
            correct = torch.eq(results[-1].argmax(dim=1), temp_labels).sum().item()
            corrects[1] += round(correct, 4)
            totals[1] += temp_labels.shape[0]

            for k in range(1, self.update_step):
                lr *= 0.95

                results = self.learner(support_x[i], params=theta)
                loss = self.cal_loss(results, support_x[i], support_y[i])

                grads = torch.autograd.grad(loss, theta, retain_graph=second_derivative_flag,
                                            create_graph=second_derivative_flag, allow_unused=True)
                theta = list(map(lambda p: p[1] - lr * p[0] if p[0] is not None else p[1], zip(grads, theta)))

                results = self.learner(query_x[i], params=theta)
                loss_after_update = self.cal_loss(results, query_x[i], query_y[i], ft_x, ft_y, clc=clc_flag)

                losses[k + 1] += loss_after_update

                with torch.no_grad():
                    temp_labels = torch.cat([query_y[i][results[0]], query_y[i][results[0]], query_y[i]],
                                            dim=0) if multi_label else query_y[i]
                    cor = torch.eq(results[-1].argmax(dim=1), temp_labels).sum().item()
                    corrects[k + 1] += round(cor, 4)
                    totals[k + 1] += temp_labels.shape[0]

        # Multi-Step Loss Optimization
        final_loss = deepcopy(losses[0])
        cur = 0.8
        for step, step_loss in enumerate(losses[0:]):
            final_loss += cur * final_loss + (1 - cur) * step_loss

        self.meta_optimizer.zero_grad()
        final_loss.backward()
        self.meta_optimizer.step()
        self.meta_scheduler.step()

        losses_output = [str(round(loss.item() / num_task, 4)) for loss in losses]
        corrects = np.array(corrects)
        totals = np.array(totals)

        print(corrects / totals)
        print(f'ave task loss: {"->".join(losses_output)}')

    def fine_tune(self, support_x, support_y, query_x, query_y, step, fine_tune=True):
        if not fine_tune:
            step = 0
        query_size = query_x.size(0)
        print(query_size)
        corrects = [0] * (step + 1)
        predictions = []
        probabilities = []

        net = deepcopy(self.learner)
        net.toggle_stage('test')
        with torch.no_grad():
            results = net(query_x)
            predictions.append(results[-1].argmax(dim=1)[-query_size:])
            correct = torch.eq(results[-1].argmax(dim=1)[-query_size:], query_y).sum().item()
            corrects[0] += correct
        probabilities.append(func.softmax(results[-1], dim=1)[-query_size:])

        params = [
            {'params': net.encoder.parameters(), 'lr': self.fine_tuning_lr},
            {'params': net.projector.parameters(), 'lr': self.fine_tuning_lr},
            {'params': net.generator.parameters(), 'lr': self.fine_tuning_lr},
            {'params': net.diagnostor.parameters(), 'lr': self.fine_tuning_lr}
        ]

        optimizer = torch.optim.AdamW(params, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        for k in tqdm(range(step)):
            net.toggle_stage('train')
            results = net(support_x)
            loss = self.cal_loss(results, support_x, support_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            net.toggle_stage('test')
            with torch.no_grad():
                _, prediction = net.test(query_x, False)
                predictions.append(prediction.argmax(dim=-1))
                cor = torch.eq(prediction.argmax(dim=-1), query_y).sum().item()
                corrects[k + 1] += cor
            probabilities.append(func.softmax(prediction.detach(), dim=-1))

        del net

        return (np.array(corrects) / query_size), predictions[-1], probabilities[-1]