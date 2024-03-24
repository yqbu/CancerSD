import numpy as np
import torch
from torch import nn
from torch.nn import functional as func
from Codes.model import ConfigModel


class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, epsilon=0.05):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets, weight=None):
        n = predictions.size()[-1]
        log_predictions = func.log_softmax(predictions, dim=-1)
        loss = -log_predictions.sum(dim=-1).mean()
        nll = func.nll_loss(log_predictions, targets, weight=weight)
        return self.epsilon * (loss / n) + (1 - self.epsilon) * nll


class ContrastiveLoss(nn.Module):

    def __init__(self, tau=0.07):
        super(ContrastiveLoss, self).__init__()
        self.tau = tau

    def forward(self, inputs):
        zi = inputs[0]
        zj = inputs[1]

        batch_size = zi.shape[0]
        assert zi.shape[0] == zj.shape[0]

        negative_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(zi.device)).float()
        representations = torch.cat([zi, zj], dim=0)
        similarities = torch.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarities, batch_size)
        sim_ji = torch.diag(similarities, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.tau)
        denominator = negative_mask * torch.exp(similarities / self.tau)

        partial_loss = -torch.log(nominator / torch.sum(denominator, dim=1))

        return torch.mean(partial_loss)


class CategoryLevelContrastive(nn.Module):

    def __init__(self, tau=0.1, sigma=0.2, start=-5, end=5, num_bins=128, in_features=512, out_features=128):
        super(CategoryLevelContrastive, self).__init__()

        self.tau = tau
        self.sigma = sigma
        self.num_bins = num_bins
        self.epsilon = 1e-8

        self.projector = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        self.bins = nn.Parameter(torch.linspace(start, end, num_bins).float(), requires_grad=True)

    def marginal_pdf(self, values):
        residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5 * (residuals / self.sigma).pow(2))

        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization
        pdf = func.softmax(pdf, dim=-1)

        return pdf, kernel_values

    def get_distribution(self, x, y):
        x = x.squeeze()
        x_norm = x / torch.norm(x, p=2, dim=1).unsqueeze(1)
        x_norm = x_norm.unsqueeze(1)

        unique_labels, _ = torch.unique(y, return_counts=True)
        distributions = []
        for label in unique_labels:
            indexes = torch.nonzero((y == label).to(torch.float)).squeeze()
            partial_embeddings = x_norm[indexes]
            batch, channel, dimension = partial_embeddings.shape
            partial_distribution, _ = self.marginal_pdf(partial_embeddings.view(batch, dimension, channel))
            partial_distribution = torch.mean(partial_distribution, dim=0).unsqueeze(0)
            distributions.append(partial_distribution)

        return torch.cat(distributions, dim=0)

    def KLD_matrix(self, batch_size, input1, input2):
        negative_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).to(input1.device).float()
        prototypes = torch.cat([input1, input2], dim=0)
        dimension1 = prototypes.unsqueeze(1)
        dimension2 = prototypes.unsqueeze(0).repeat(prototypes.shape[0], 1, 1)

        intermediate = dimension1 * (dimension1.log() - dimension2.log())
        intermediate = intermediate.sum(dim=-1)

        return negative_mask * intermediate

    def forward(self, input_embeddings_1, y1, input_embeddings_2, y2):
        distribution_A = self.get_distribution(self.projector(input_embeddings_1).unsqueeze(1), y1)
        distribution_B = self.get_distribution(self.projector(input_embeddings_2).unsqueeze(1), y2)

        M = 0.5 * (distribution_A + distribution_B)
        batch_size = M.shape[0]

        JSD_matrix = 0.5 * (self.KLD_matrix(batch_size, distribution_A, M) +
                            self.KLD_matrix(batch_size, distribution_B, M))

        div_ij = torch.diag(JSD_matrix, batch_size)
        div_ji = torch.diag(JSD_matrix, -batch_size)
        positives = torch.cat([div_ij, div_ji], dim=0)

        nominator = torch.exp(positives / self.tau)
        denominator = torch.exp(JSD_matrix / self.tau)

        partial_loss = -torch.log(nominator / torch.sum(denominator, dim=1))

        return torch.mean(partial_loss)


class BaseLoss(ConfigModel):

    def __init__(self, task_num=None, coefficient=None):
        super(BaseLoss, self).__init__()
        if task_num is not None:
            self.task_num = task_num
            coefficient = [1] * self.task_num
        else:
            self.task_num = len(coefficient)
        self.config = [('coefficient', np.array([coefficient]))]
        self.weight = self.create_parameters_list()

    def forward(self, losses, params=None):
        assert losses.shape[-1] == self.task_num
        if params is None:
            params = self.parameters()
        return self.calculate_by_config(losses, params)

    def parameters(self):
        return self.weight
