from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as func
from tqdm import tqdm

from cancersd.utils.enhancement import get_omics_masking
from cancersd.losses.loss import ContrastiveLoss, CategoryLevelContrastive
from cancersd.utils.common import deepcopy


class ConfigModel(nn.Module):

    def __init__(self, config=None):
        super(ConfigModel, self).__init__()
        self.config = config
        self.stage = None
        self.toggle_stage('train')

    def toggle_stage(self, stage):
        self.stage = stage
        if stage == 'train':
            self.train()
        elif stage == 'test':
            self.eval()

    def create_parameters_list(self, config=None):
        if config is None:
            config = self.config

        parameters = nn.ParameterList()
        for (name, param) in config:
            if name == 'linear':
                w = nn.Parameter(torch.ones(*param))
                nn.init.xavier_normal_(w)
                parameters.append(w)
                parameters.append(nn.Parameter(torch.zeros(param[0])))
            if name == 'nonbias_linear':
                w = nn.Parameter(torch.ones(*param))
                nn.init.xavier_normal_(w)
                parameters.append(w)
            elif name == 'coefficient':
                coe = nn.Parameter(torch.from_numpy(param).float())
                parameters.append(coe)
            elif name == 'weight':
                w = nn.Parameter(torch.ones(*param))
                nn.init.xavier_normal_(w)
                parameters.append(w)
            elif name == 'bias':
                b = nn.Parameter(torch.zeros(*param))
                parameters.append(b)
            elif name == 'prelu':
                w = nn.Parameter(torch.tensor(0.25 if len(param) == 0 else float(param[0])))
                parameters.append(w)
            elif name == 'attention':
                attention = torch.ones(*param)
                attention = nn.Parameter(attention)
                parameters.append(attention)
            elif name == 'batch_norm':
                shape = param[0]
                gamma = nn.Parameter(torch.ones(shape))
                beta = nn.Parameter(torch.zeros(shape))
                moving_mean = torch.zeros(shape)
                moving_var = torch.ones(shape)

                parameters.append(gamma)
                parameters.append(beta)
                parameters.append(moving_mean)
                parameters.append(moving_var)

        return parameters

    def calculate_by_config(self, x, params):
        index = 0
        original_x = torch.clone(x)
        for name, param in self.config:
            if name == 'linear':
                w, b = params[index], params[index + 1]
                x = func.linear(x, w, b)
                index += 2
            elif name == 'nonbias_linear':
                w = params[index]
                x = func.linear(x, w)
                index += 1
            elif name == 'coefficient':
                coe = params[index]
                x = torch.sum(coe * x)
                index += 1
            elif name == 'dropout':
                rate = param[0] if len(param) else 0.1
                if self.stage == 'train':
                    x = func.dropout(x, p=rate)
            elif name == 'attention':
                attention = params[index]
                attention = torch.tanh(attention)
                x = attention.mul(x)
                index += 1
            elif name == 'residual_start':
                original_x = torch.clone(x)
            elif name == 'residual_end':
                x += original_x
            elif name == 'mish':
                x = func.mish(x)
            elif name == 'relu':
                x = func.relu(x)
            elif name == 'lrelu':
                rate = param[0] if len(param) else 0.01
                x = func.leaky_relu(x, rate)
            elif name == 'elu':
                x = func.elu(x, 1)
            elif name == 'prelu':
                p = params[index]
                x = func.prelu(x, p)
                index += 1
            elif name == 'gelu':
                x = func.gelu(x)
            elif name == 'tanh':
                x = torch.tanh(x)
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
            elif name == 'bn':
                mean = x.mean(dim=0)
                std = x.std(dim=0)
                x = (x - mean) / std
            elif name == 'batch_norm':
                eps = 1e-5
                momentum = 0.9

                gamma = params[index]
                beta = params[index + 1]
                moving_mean = params[index + 2]
                moving_var = params[index + 3]

                if moving_mean.device != x.device:
                    moving_mean = moving_mean.to(x.device)
                    moving_var = moving_var.to(x.device)

                if self.stage == 'test':
                    x = (x - moving_mean) / torch.sqrt(moving_var + eps)
                elif self.stage == 'train':
                    mean = x.mean(dim=0)
                    var = x.var(dim=0)
                    x = (x - mean) / torch.sqrt((var + eps))

                    params[index + 2] = momentum * moving_mean + (1.0 - momentum) * mean
                    params[index + 3] = momentum * moving_var + (1.0 - momentum) * var

                x = gamma * x + beta
                index += 4

        assert index == len(params)

        return x

    def get_parameters_section(self):
        names = []
        lengths = []
        for name, module in self.named_children():
            names.append(name)
            if isinstance(module, nn.ParameterList):
                lengths.append(len(module))
            elif isinstance(module, nn.ModuleList):
                lengths.append(len(module))
            else:
                lengths.append(len(module.parameters()))

        current = 0
        param_length_list = [current]
        for i in range(len(lengths)):
            current += lengths[i]
            param_length_list.append(current)

        parameter_section = {}
        for i in range(len(lengths)):
            parameter_section[names[i]] = [param_length_list[i], param_length_list[i + 1]]

        return parameter_section

    def print_model(self):
        models = {}
        for name, module in self.named_modules():
            if isinstance(module, ConfigModel):
                if module.config is not None:
                    models[name] = module.config

        return models


class Encoder(ConfigModel):

    def __init__(self, input_dimension, output_dimension, config=None):
        super(Encoder, self).__init__(config)
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.config = config

        self.local = self.create_parameters_list()

    def forward(self, feature, params=None):
        return self.calculate_by_config(feature, self.local if params is None else params)

    def parameters(self):
        return self.local


class Decoder(ConfigModel):

    def __init__(self, input_dimension, output_dimension, config=None):
        super(Decoder, self).__init__(config)
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.config = config

        self.local = self.create_parameters_list()

    def forward(self, feature, params=None):
        return self.calculate_by_config(feature, self.local if params is None else params)

    def parameters(self):
        return self.local


class PatientEncoder(ConfigModel):

    def __init__(self, omics_dimensions_dict, embedding_dimension, rank):
        super(PatientEncoder, self).__init__()
        self.omics_types = list(omics_dimensions_dict.keys())
        self.omics_dimensions_dict = omics_dimensions_dict
        self.embedding_dimension = embedding_dimension
        hidden_dimension = 768
        self.rank = rank

        self.config = [('weight', [self.rank, hidden_dimension + 1, embedding_dimension])] * len(omics_dimensions_dict)
        self.config += [
            ('weight', [1, self.rank]),
            ('bias', [1, embedding_dimension])
        ]
        paramLens = []

        self.encoders = {}
        if 'methylation' in self.omics_types:
            self.methylation_encoder = Encoder(omics_dimensions_dict['methylation'], embedding_dimension, [
                ('linear', [hidden_dimension, omics_dimensions_dict['methylation']]),
                ('prelu', []),
            ])
            self.encoders['methylation'] = self.methylation_encoder
            paramLens.append(len(self.methylation_encoder.parameters()))
        if 'miRNA' in self.omics_types:
            self.miRNA_encoder = Encoder(omics_dimensions_dict['miRNA'], embedding_dimension, [
                ('linear', [hidden_dimension, omics_dimensions_dict['miRNA']]),
                ('prelu', []),
            ])
            self.encoders['miRNA'] = self.miRNA_encoder
            paramLens.append(len(self.miRNA_encoder.parameters()))
        if 'mRNA' in self.omics_types:
            self.mRNA_encoder = Encoder(omics_dimensions_dict['mRNA'], embedding_dimension, [
                ('linear', [hidden_dimension, omics_dimensions_dict['mRNA']]),
                ('prelu', []),
            ])
            self.encoders['mRNA'] = self.mRNA_encoder
            paramLens.append(len(self.mRNA_encoder.parameters()))
        self.local = self.create_parameters_list()
        self.parameters_section = self.get_parameters_section()

    def forward(self, patient, params=None):
        if params is None:
            params = self.parameters()
        omics = torch.split(patient, list(self.omics_dimensions_dict.values()), dim=-1)
        masks = [(omic.sum(dim=1) > 0).float().unsqueeze(-1).repeat((1, self.embedding_dimension))
                 for omic in omics]
        embeddings = self.encode(patient, params)

        return self.integrate(patient.shape[0], embeddings, masks, params)

    def encode(self, patient, params=None):
        if params is None:
            params = self.parameters()
        omics = torch.split(patient, list(self.omics_dimensions_dict.values()), dim=-1)

        embeddings = []
        for i, moleType in enumerate(self.omics_types):
            start, end = self.parameters_section[moleType + '_encoder']
            embedding = self.encoders[moleType](omics[i], params[start:end])
            embeddings.append(embedding)

        return embeddings

    def integrate(self, sample_size, multi_omics_embeddings, masks=None, params=None):
        if params is None:
            params = self.parameters()
        fusion_start = self.parameters_section['local'][0]
        weight_position = fusion_start + len(self.omics_types)
        bias_position = weight_position + 1
        device = multi_omics_embeddings[0].device

        integrations = []
        for i, moleType in enumerate(self.omics_types):
            embedding = torch.cat([multi_omics_embeddings[i], torch.ones(sample_size, 1, device=device)], dim=1)
            integrations.append(torch.matmul(embedding, params[fusion_start + i]))

        integrations = [torch.matmul(params[weight_position], fusion.permute(1, 0, 2)).squeeze()
                        for fusion in integrations]
        if masks is not None:
            integrations = [fusion * masks[i] + (1. - masks[i]) for i, fusion in enumerate(integrations)]
        integrated_omics = torch.ones((sample_size, self.embedding_dimension), device=device)
        for i in range(len(self.omics_types)):
            integrated_omics *= integrations[i]

        return integrated_omics + params[bias_position]

    def parameters(self):
        parameters = nn.ParameterList()
        if 'methylation' in self.omics_types:
            parameters += self.methylation_encoder.parameters()
        if 'miRNA' in self.omics_types:
            parameters += self.miRNA_encoder.parameters()
        if 'mRNA' in self.omics_types:
            parameters += self.mRNA_encoder.parameters()
        parameters += self.local

        return parameters


class Generator(ConfigModel):

    def __init__(self, omics_dimensions_dict, embedding_dimension):
        super(Generator, self).__init__()
        self.omics_dimensions_dict = omics_dimensions_dict
        self.omics_types = list(omics_dimensions_dict.keys())
        self.embedding_dimension = embedding_dimension

        lengths = []
        self.decoders = {}
        if 'methylation' in self.omics_types:
            self.methylation_decoder = Decoder(omics_dimensions_dict['methylation'], embedding_dimension, [
                ('attention', [embedding_dimension]),
                ('linear', [embedding_dimension * 2, embedding_dimension]),
                ('prelu', []),
                ('linear', [embedding_dimension * 4, embedding_dimension * 2]),
                ('prelu', []),
                ('linear', [omics_dimensions_dict['methylation'], embedding_dimension * 4]),
                ('sigmoid', []),
            ])
            self.decoders['methylation'] = self.methylation_decoder
            lengths.append(len(self.methylation_decoder.parameters()))
        if 'miRNA' in self.omics_types:
            self.miRNA_decoder = Decoder(omics_dimensions_dict['miRNA'], embedding_dimension, [
                ('attention', [embedding_dimension]),
                ('linear', [embedding_dimension, embedding_dimension]),
                ('prelu', []),
                ('linear', [embedding_dimension, embedding_dimension]),
                ('prelu', []),
                ('linear', [omics_dimensions_dict['miRNA'], embedding_dimension]),
                ('sigmoid', []),
            ])
            self.decoders['miRNA'] = self.miRNA_decoder
            lengths.append(len(self.miRNA_decoder.parameters()))
        if 'mRNA' in self.omics_types:
            self.mRNA_decoder = Decoder(omics_dimensions_dict['mRNA'], embedding_dimension, [
                ('attention', [embedding_dimension]),
                ('linear', [embedding_dimension * 2, embedding_dimension]),
                ('prelu', []),
                ('linear', [embedding_dimension * 4, embedding_dimension * 2]),
                ('prelu', []),
                ('linear', [omics_dimensions_dict['mRNA'], embedding_dimension * 4]),
                ('sigmoid', []),
            ])
            self.decoders['mRNA'] = self.mRNA_decoder
            lengths.append(len(self.mRNA_decoder.parameters()))

        self.parameters_section = self.get_parameters_section()

    def forward(self, latent, params=None):
        if params is None:
            params = self.parameters()

        reconstructed = []
        for i, moleType in enumerate(self.omics_types):
            start, end = self.parameters_section[moleType + '_decoder']
            reconstructed.append(self.decoders[moleType](latent, params[start:end]))

        return torch.cat(reconstructed, dim=1)

    def parameters(self):
        parameters = nn.ParameterList()
        if 'methylation' in self.omics_types:
            parameters += self.methylation_decoder.parameters()
        if 'miRNA' in self.omics_types:
            parameters += self.miRNA_decoder.parameters()
        if 'mRNA' in self.omics_types:
            parameters += self.mRNA_decoder.parameters()

        return parameters


@dataclass(frozen=True)
class ModelOutput:
    complete_index: torch.Tensor
    incomplete_index: torch.Tensor
    available: torch.Tensor
    origins: torch.Tensor
    projections: tuple[torch.Tensor, torch.Tensor] | torch.Tensor
    reconstructed: torch.Tensor
    generated: torch.Tensor
    diagnoses: torch.Tensor

    def as_tuple(self) -> tuple:
        return (
            self.complete_index,
            self.incomplete_index,
            self.available,
            self.origins,
            self.projections,
            self.reconstructed,
            self.generated,
            self.diagnoses,
        )

    def __iter__(self) -> Iterator:
        return iter(self.as_tuple())

    def __getitem__(self, index: int):
        return self.as_tuple()[index]


class CancerSD(ConfigModel):

    def __init__(self, num_way, omics_dimensions_dict, embedding_dimension, rank):
        super(CancerSD, self).__init__()
        self.num_way = num_way
        self.omics_dimensions_dict = omics_dimensions_dict
        self.embedding_dimension = embedding_dimension
        self.rank = rank

        projection_dimension = 64
        self.encoder = PatientEncoder(omics_dimensions_dict, embedding_dimension, rank)

        self.projector = Encoder(embedding_dimension, projection_dimension, [
            ('nonbias_linear', [embedding_dimension, embedding_dimension]),
            ('batch_norm', [embedding_dimension]),
            ('relu', []),
            ('nonbias_linear', [embedding_dimension // 2, embedding_dimension]),
            ('batch_norm', [embedding_dimension // 2]),
            ('relu', []),
            ('nonbias_linear', [projection_dimension, embedding_dimension // 2]),
            ('batch_norm', [projection_dimension]),
        ])
        self.generator = Generator(omics_dimensions_dict, embedding_dimension)

        self.diagnostor = Encoder(embedding_dimension, num_way, [
            ('linear', [embedding_dimension // 2, embedding_dimension]),
            ('prelu', []),
            ('linear', [embedding_dimension // 2, embedding_dimension // 2]),
            ('prelu', []),
            ('dropout', [0.1]),
            ('linear', [num_way, embedding_dimension // 2]),
        ])

        self.parameters_section = self.get_parameters_section()

    def _prepare_inputs(self, patients: torch.Tensor) -> tuple[torch.Tensor, ...]:
        available = (~torch.isnan(patients)).long()
        origins = torch.nan_to_num(patients, nan=0.0)
        missing = 1 - available

        return available, origins, missing

    def _split_complete_incomplete(self, patients: torch.Tensor) -> tuple[torch.Tensor, ...]:
        complete_index = torch.nonzero(~torch.isnan(patients).any(dim=1), as_tuple=False).reshape(-1)
        total_indices = torch.arange(patients.size(0), device=patients.device)
        incomplete_indices = total_indices[~torch.isin(total_indices, complete_index)]

        return complete_index, incomplete_indices

    def _forward_contrastive_task(self, complete_patients: torch.Tensor, params):
        encoder_start, encoder_end = self.parameters_section['encoder']
        projector_start, projector_end = self.parameters_section['projector']

        augmented_view = get_omics_masking(complete_patients, self.omics_dimensions_dict)
        augmented_view_apostrophe = get_omics_masking(complete_patients, self.omics_dimensions_dict)

        embeddings = self.encoder(augmented_view, params[encoder_start:encoder_end])
        embeddings_apostrophe = self.encoder(augmented_view_apostrophe, params[encoder_start:encoder_end])
        contrastive_embeddings = (embeddings, embeddings_apostrophe)

        projection = self.projector(embeddings, params[projector_start:projector_end])
        projection_apostrophe = self.projector(embeddings_apostrophe, params[projector_start:projector_end])
        projections = (projection, projection_apostrophe)

        return contrastive_embeddings, projections

    def _forward_reconstruction_task(self, complete_patients: torch.Tensor, params):
        encoder_start, encoder_end = self.parameters_section['encoder']
        generator_start, generator_end = self.parameters_section['generator']

        patients_cloze = get_omics_masking(complete_patients, self.omics_dimensions_dict)
        embeddings_cloze = self.encoder(patients_cloze, params[encoder_start:encoder_end])
        reconstructed_cloze = self.generator(embeddings_cloze, params[generator_start:generator_end])

        return reconstructed_cloze

    def _forward_diagnosis_task(
            self, features: torch.Tensor, missing: torch.Tensor,
            contrastive_embeddings: tuple[torch.Tensor, torch.Tensor], params
    ):
        encoder_start, encoder_end = self.parameters_section['encoder']
        generator_start, generator_end = self.parameters_section['generator']
        diagnostor_start, diagnostor_end = self.parameters_section['diagnostor']

        latents = self.encoder(features, params[encoder_start:encoder_end])
        generated = self.generator(latents, params[generator_start:generator_end])
        fusions = features + missing * generated.detach()

        final_embeddings = self.encoder(fusions, params[encoder_start:encoder_end])
        embeddings, embeddings_apostrophe = contrastive_embeddings
        final_embeddings = torch.cat([embeddings, embeddings_apostrophe, final_embeddings], dim=0)
        diagnoses = self.diagnostor(final_embeddings, params[diagnostor_start:diagnostor_end])

        return generated, diagnoses

    def forward(self, patients, params=None):
        if len(self.omics_dimensions_dict) == 1:
            return self.single_omics_forward(patients)

        if params is None:
            params = self.parameters()

        available, origins, missing = self._prepare_inputs(patients)
        complete_index, incomplete_index = self._split_complete_incomplete(patients)

        complete_patients = origins[complete_index]
        if complete_patients.dim() == 1:
            complete_patients = complete_patients.unsqueeze(0)

        # contrastive learning tasks
        contrastive_embeddings, projections = self._forward_contrastive_task(complete_patients, params)

        # masking-and-reconstruction tasks
        reconstructed_cloze = self._forward_reconstruction_task(complete_patients, params)

        # diagnosis task
        generated, diagnoses = self._forward_diagnosis_task(origins, missing, contrastive_embeddings, params)

        return ModelOutput(
            complete_index=complete_index,
            incomplete_index=incomplete_index,
            available=available,
            origins=origins,
            projections=projections,
            reconstructed=reconstructed_cloze,
            generated=generated,
            diagnoses=diagnoses
        )

    def test(self, patients, print_each_sim=True):
        if len(self.omics_dimensions_dict) == 1:
            return self.single_omics_test(patients, print_each_sim)

        available, origins, missing = self._prepare_inputs(patients)

        latents = self.encoder(origins)
        generated = self.generator(latents)
        fusions = origins + missing * generated

        embeddings = self.encoder(fusions)
        diagnoses = self.diagnostor(embeddings)

        completions = generated * available

        if print_each_sim:
            omics_dimensions = list(self.omics_dimensions_dict.values())
            split_completion = torch.split(completions, omics_dimensions, dim=1)
            split_original = torch.split(origins, omics_dimensions, dim=1)

            similarities = []
            for i in range(len(omics_dimensions)):
                similarity = torch.cosine_similarity(split_completion[i], split_original[i]).unsqueeze(1)
                similarities.append(similarity)
            similarities = torch.cat(similarities, dim=1)
            mean_similarities = torch.mean(similarities, dim=0).detach().cpu().numpy()
            std_similarities = torch.std(similarities, dim=0).detach().cpu().numpy()
            for i, omics in enumerate(self.omics_dimensions_dict.keys()):
                print(f'average similarity of {omics}: {mean_similarities[i]:.3f}, std: {std_similarities[i]:.3f}')

        return torch.cosine_similarity(completions, origins), diagnoses

    def single_omics_forward(self, patients, params=None):
        if params is None:
            params = self.parameters()
        encoder_start, encoder_end = self.parameters_section['encoder']
        generator_start, generator_end = self.parameters_section['generator']
        diagnostor_start, diagnostor_end = self.parameters_section['diagnostor']

        available, _, _ = self._prepare_inputs(patients)

        # reconstruction tasks
        embeddings_cloze = self.encoder(patients, params[encoder_start:encoder_end])
        reconstructed_cloze = self.generator(embeddings_cloze, params[generator_start:generator_end])

        # diagnosis task
        final_embeddings = self.encoder(patients, params[encoder_start:encoder_end])
        diagnoses = self.diagnostor(final_embeddings, params[diagnostor_start:diagnostor_end])

        complete_index = torch.tensor([])
        incomplete_index = torch.tensor([])
        projections = torch.tensor([])

        # return complete_index, incomplete_index, available, patients, \
        #     projections, reconstructed_cloze, reconstructed_cloze, diagnoses
        return ModelOutput(
            complete_index=complete_index,
            incomplete_index=incomplete_index,
            available=available,
            origins=patients,
            projections=projections,
            reconstructed=reconstructed_cloze,
            generated=reconstructed_cloze,
            diagnoses=diagnoses
        )

    def single_omics_test(self, patients, print_each_sim=False):
        latents = self.encoder(patients)
        reconstruction = self.generator(latents)
        diagnoses = self.diagnostor(latents)

        similarity = torch.cosine_similarity(reconstruction, patients)

        if print_each_sim:
            print(f'average similarity: {similarity.mean():.3f}, std: {similarity.std():.3f}')

        return similarity, diagnoses

    def impute(self, patients):
        available, origins, missing = self._prepare_inputs(patients)

        latents = self.encoder(origins)
        generated = self.generator(latents)
        fusions = origins + missing * generated

        return fusions

    def encode(self, patients, impute=True):
        origins = torch.where(torch.isnan(patients), torch.full_like(patients, 0), patients)

        return self.encoder(self.impute(patients)) if impute else self.encoder(origins)

    def partial_encode(self, patients):
        imputed_patients = self.impute(patients)
        omics_embeddings = self.encoder.encode(imputed_patients)
        fusion_embeddings = self.encode(imputed_patients)

        return omics_embeddings, fusion_embeddings

    def parameters(self):
        parameters = nn.ParameterList()
        parameters += self.encoder.parameters()
        parameters += self.projector.parameters()
        parameters += self.generator.parameters()
        parameters += self.diagnostor.parameters()

        return parameters
