import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from Codes.utils import cumulative_sum


class BaseDataset(Dataset):

    def __init__(self):
        super(BaseDataset, self).__init__()

    def __getitem__(self, index):
        pass

    def random_split(self, ratios):
        train_set = []
        test_set = []
        for subtype in self.subtype_number_dict.keys():
            temp_df = self.patients_info[self.patients_info['subtype'] == subtype]
            disorder = torch.randperm(temp_df.shape[0])

            train_size = round(temp_df.shape[0] * ratios[0])
            train_index = disorder[:train_size]
            test_index = disorder[train_size:]

            train_set.append(temp_df.iloc[train_index, :])
            test_set.append(temp_df.iloc[test_index, :])

        train_set = pd.concat(train_set, axis=0)
        test_set = pd.concat(test_set, axis=0)

        base_setting = {
            'omics_types': self.omics_types,
            'omics_dimensions': self.omics_dimensions,
            'omics_dimensions_dict': self.omics_dimensions_dict,
            'subtype_number_dict': self.subtype_number_dict
        }

        train = PatientDataset(**base_setting, patients_info=train_set)
        test = PatientDataset(**base_setting, patients_info=test_set)

        return train, test

    def train_validation_test(self, ratios):
        train_set = []
        validation_set = []
        test_set = []
        for subtype in self.subtype_number_dict.keys():
            temp_df = self.patients_info[self.patients_info['subtype'] == subtype]
            disorder = torch.randperm(temp_df.shape[0])

            train_size = round(temp_df.shape[0] * ratios[0])
            validation_size = round(temp_df.shape[0] * ratios[1])

            train_index = disorder[:train_size]
            validation_index = disorder[train_size:train_size + validation_size]
            test_index = disorder[train_size + validation_size:]

            train_set.append(temp_df.iloc[train_index, :])
            validation_set.append(temp_df.iloc[validation_index, :])
            test_set.append(temp_df.iloc[test_index, :])
        train_set = pd.concat(train_set, axis=0)
        validation_set = pd.concat(validation_set, axis=0)
        test_set = pd.concat(test_set, axis=0)

        base_setting = {
            'omics_types': self.omics_types,
            'omics_dimensions': self.omics_dimensions,
            'omics_dimensions_dict': self.omics_dimensions_dict,
            'subtype_number_dict': self.subtype_number_dict
        }

        train = PatientDataset(**base_setting, patients_info=train_set)
        validation = PatientDataset(**base_setting, patients_info=validation_set)
        test = PatientDataset(**base_setting, patients_info=test_set)

        return train, validation, test

    def getNwayKshot(self, shot):
        support_index = []
        query_index = []
        for subtype in self.subtype_number_dict.keys():
            temp_df = self.patients_info[self.patients_info['subtype'] == subtype]
            disorder = torch.randperm(temp_df.shape[0])
            sample_list = disorder[:shot]
            support_index += list(temp_df.iloc[sample_list, :1].index.values)

            not_sampled_list = disorder[shot:]
            query_index += list(temp_df.iloc[not_sampled_list, :1].index.values)

        base_setting = {
            'omics_types': self.omics_types,
            'omics_dimensions': self.omics_dimensions,
            'omics_dimensions_dict': self.omics_dimensions_dict,
            'subtype_number_dict': self.subtype_number_dict
        }

        train = PatientDataset(**base_setting, patients_info=self.patients_info.loc[support_index])
        test = PatientDataset(**base_setting, patients_info=self.patients_info.loc[query_index])

        return train, test

    def get_class_weight(self, ratio):
        num_class = len(self.subtype_number_dict)
        num_sample = self.patients_info.shape[0]
        class_weight = [0] * num_class

        for k, v in ratio.items():
            class_weight[self.subtype_number_dict[k]] = num_sample / (v * num_class)

        return class_weight

    def adjust_omics_types(self, omics_types):
        print(f'omics types from {self.omics_types} to {omics_types}')
        assert len(self.omics_types) >= len(omics_types)
        omics_sections = cumulative_sum(self.omics_dimensions)

        new_patient_info = []
        new_dict = {}
        for i, mole in enumerate(self.omics_dimensions_dict.keys()):
            temp_data = self.patients_info.iloc[:, omics_sections[i]:omics_sections[i + 1]]
            if mole in omics_types:
                new_patient_info.append(temp_data)
                new_dict[mole] = self.omics_dimensions_dict[mole]

        new_patient_info.append(self.patients_info.iloc[:, -1])
        new_patient_info = pd.concat(new_patient_info, axis=1)

        base_setting = {
            'omics_types': omics_types,
            'omics_dimensions': list(new_dict.values()),
            'omics_dimensions_dict': new_dict,
            'subtype_number_dict': self.subtype_number_dict
        }

        return PatientDataset(**base_setting, patients_info=new_patient_info)


class PatientDataset(BaseDataset):

    def __init__(self, omics_types, **kwargs):
        super(PatientDataset, self).__init__()
        self.omics_types = omics_types
        cancer = kwargs.get('cancer', 'STAD')
        root = kwargs.get('root', '../Data')
        if isinstance(cancer, list):
            paths = [os.path.join(root, c) for c in cancer]
        elif isinstance(cancer, str):
            paths = os.path.join(root, cancer)
        diagnosis_folder = kwargs.get('diagnosis_folder', 'patient_diagnose.csv')

        if kwargs.get('patients_info') is None:
            self.omics_dimensions, self.omics_dimensions_dict, self.patients_info = self.load_data(paths,
                                                                                                   diagnosis_folder,
                                                                                                   kwargs)
            self.patients_info = self.patients_info.sort_index(axis=0, ascending=True)
        else:
            self.omics_dimensions = kwargs.get('omics_dimensions')
            self.omics_dimensions_dict = kwargs.get('omics_dimensions_dict')
            self.patients_info = kwargs.get('patients_info')

        class_ratio = self.patients_info['subtype'].value_counts().to_dict()
        self.num_subtype = len(class_ratio)
        if kwargs.get('subtype_number_dict') is None:
            self.subtype_number_dict = dict(zip(class_ratio.keys(), range(self.num_subtype)))
        else:
            self.subtype_number_dict = kwargs.get('subtype_number_dict')
        self.number_subtype_dict = dict(zip(self.subtype_number_dict.values(), self.subtype_number_dict.keys()))

        self.class_weight = self.get_class_weight(class_ratio)

    def load_data(self, root, diagnosis_folder, dicts):
        omics_data_list = []
        patients_with_incomplete = {}
        patients_with_incomplete_list = []
        omics_dimensions = []
        suffix = dicts.get('suffix', '')

        for moleType in self.omics_types:
            if os.path.exists(root + '/patient_lack_' + moleType + '.npy'):
                patients_with_incomplete[moleType] = list(np.load(root + '/patient_lack_' + moleType + '.npy'))
                patients_with_incomplete_list += patients_with_incomplete[moleType]
            path = os.path.join(root, moleType + suffix + '.csv')
            omics_data = pd.read_csv(path).set_index([moleType]).T
            omics_data_list.append(omics_data)
            omics_dimensions.append(omics_data.shape[1])

        omics_data_list = pd.concat(omics_data_list, axis=1)
        patients_with_incomplete_list = list(set(patients_with_incomplete_list))

        diagnoses = pd.read_csv(os.path.join(root, diagnosis_folder)).set_index(['aspect'])
        diagnoses = diagnoses[omics_data_list.index].T
        if dicts.get('race') is not None:
            mode = 'include' if dicts.get('mode') is None else dicts.get('mode')
            if mode == 'include':
                diagnoses = diagnoses[diagnoses['race'] == dicts.get('race')]
            elif mode == 'exclude':
                diagnoses = diagnoses[diagnoses['race'] != dicts.get('race')]
        drop_columns = dicts.get('diag_drop_columns', ['stage', 'race'])
        diagnoses.drop(columns=drop_columns, inplace=True)
        omics_data_list = omics_data_list.loc[diagnoses.index, :]

        patients_reserved = []
        state = dicts.get('omics_state', 'both')
        if state == 'complete':
            patients_reserved = list(set(omics_data_list.index.values) - set(patients_with_incomplete_list))
        elif state == 'missing':
            patients_reserved = list(set(omics_data_list.index.values) & set(patients_with_incomplete_list))
        elif state == 'both':
            patients_reserved = omics_data_list.index.values

        sections = cumulative_sum(omics_dimensions)
        temp_omics = dict()
        for i, moleType in enumerate(self.omics_types):
            temp_omics[moleType] = omics_data_list.iloc[:, sections[i]:sections[i + 1]]

        omics_dimensions_dict = {}
        omics_reserved = []
        for moleType in self.omics_types:
            temp_omics[moleType] = temp_omics[moleType].sort_index(axis=0)
            temp_omics[moleType] = temp_omics[moleType].sort_index(axis=1)
            omics_reserved.append(temp_omics[moleType])
            omics_dimensions_dict[moleType] = temp_omics[moleType].shape[1]

        omics_reserved = pd.concat(omics_reserved, axis=1)

        omics_data_list = omics_reserved.loc[patients_reserved, :]
        diagnoses = diagnoses.loc[patients_reserved, :]
        diagnoses = diagnoses.sort_index(axis=0)

        patients_info = pd.concat([omics_data_list, diagnoses], axis=1)

        return omics_dimensions, omics_dimensions_dict, patients_info

    def get_pair(self, shot):
        samples = []
        for subtype in self.subtype_number_dict.keys():
            temp_df = self.patients_info[self.patients_info['subtype'] == subtype]
            disorder = torch.randperm(temp_df.shape[0])
            samples.append(temp_df.iloc[disorder[:shot], :])

        samples = pd.concat(samples, axis=0)

        disorder = torch.randperm(samples.shape[0])
        patients = samples.iloc[disorder, :-1].to_numpy()
        labels = np.array([self.subtype_number_dict.get(label) for label in samples.iloc[:, -1]])[disorder]

        return torch.tensor(patients, dtype=torch.float32), torch.from_numpy(labels)

    def get_batch(self, batch_sz, device=None):
        disorder = torch.randperm(self.patients_info.shape[0])[:batch_sz]
        samples = self.patients_info.iloc[disorder, :]
        patients = torch.tensor(samples.iloc[:, :-1].to_numpy(), dtype=torch.float32)
        labels = torch.from_numpy(np.array([self.subtype_number_dict.get(label) for label in samples.iloc[:, -1]]))

        if device is not None:
            patients = patients.to(device)
            labels = labels.to(device)

        return patients, labels

    def get_mat(self):
        patients_info = self.patients_info
        patients = patients_info.iloc[:, :-1].to_numpy()
        labels = np.array([self.subtype_number_dict.get(label) for label in patients_info.iloc[:, -1]])

        return patients, labels

    def extend_as(self, dataset):
        omics_dimensions_dict = dataset.omics_dimensions_dict
        omics_dimensions = dataset.omics_dimensions
        old_section = cumulative_sum(self.omics_dimensions)
        section = cumulative_sum(omics_dimensions)

        assert len(self.omics_types) <= len(omics_dimensions)

        new_patient_info = []
        for i, mole in enumerate(omics_dimensions_dict.keys()):
            if mole in self.omics_types:
                new_patient_info.append(self.patients_info.iloc[:, old_section[0]:old_section[1]])
                continue
            molecules = dataset.patients_info.columns.values[section[i]:section[i + 1]]
            temp_info = pd.DataFrame(index=self.patients_info.index, columns=molecules)
            new_patient_info.append(temp_info)
        new_patient_info.append(self.patients_info.iloc[:, -1])
        new_patient_info = pd.concat(new_patient_info, axis=1)

        base_setting = {
            'omics_types': list(omics_dimensions_dict.keys()),
            'omics_dimensions': omics_dimensions,
            'omics_dimensions_dict': omics_dimensions_dict,
            'subtype_number_dict': self.subtype_number_dict
        }

        return PatientDataset(**base_setting, patients_info=new_patient_info)

    def __getitem__(self, index):
        patientData = self.patients_info.iloc[index, :-1].to_list()
        patientLabel = self.subtype_number_dict.get(self.patients_info.iloc[index, -1])

        return torch.tensor(patientData, dtype=torch.float32), torch.tensor(patientLabel)

    def __len__(self):
        return self.patients_info.shape[0]


class ConcatDataset(BaseDataset):

    def __init__(self, datasets, subtype_number_dict=None):
        super(ConcatDataset, self).__init__()
        self.omics_types = datasets[0].omics_types
        patients_info = []
        for dataset in datasets:
            patients_info.append(dataset.patients_info)

        self.patients_info = pd.concat(patients_info, axis=0)
        self.omics_dimensions = datasets[0].omics_dimensions
        self.omics_dimensions_dict = datasets[0].omics_dimensions_dict

        num_label = self.patients_info['subtype'].value_counts().to_dict()
        self.num_subtype = len(num_label)
        if subtype_number_dict is None:
            self.subtype_number_dict = datasets[0].subtype_number_dict
        else:
            self.subtype_number_dict = subtype_number_dict
        self.number_subtype_dict = dict(zip(self.subtype_number_dict.values(), self.subtype_number_dict.keys()))

        class_ratio = self.patients_info['subtype'].value_counts().to_dict()
        self.class_weight = self.get_class_weight(class_ratio)

    def __len__(self):
        return self.patients_info.shape[0]

    def __getitem__(self, index):
        patientData = self.patients_info.iloc[index, :-1].to_list()
        patientLabel = self.subtype_number_dict.get(self.patients_info.iloc[index, -1])

        return torch.tensor(patientData, dtype=torch.float32), torch.tensor(patientLabel)


class MetaTaskDataset(Dataset):

    def __init__(self, omics_types, num_way, num_shot, **kwargs):
        super(MetaTaskDataset, self).__init__()
        self.omics_types = omics_types
        self.num_way = num_way
        self.num_shot = num_shot

        self.omics_dimensions = kwargs.get('omics_dimensions')
        self.patients_info = kwargs.get('patients_info')

        num_label = self.patients_info['subtype'].value_counts().to_dict()

        if kwargs.get('subtype_number_dict') is None:
            self.subtype_number_dict = dict(zip(num_label.keys(), range(self.num_way)))
        else:
            self.subtype_number_dict = kwargs.get('subtype_number_dict')

        self.number_subtype_dict = dict(zip(self.subtype_number_dict.values(), self.subtype_number_dict.keys()))

    def __getitem__(self, index):
        patientDatas = []
        patientLabels = []
        queryX = []
        queryY = []
        for subtype in self.subtype_number_dict.keys():
            temp_df = self.patients_info[self.patients_info['subtype'] == subtype]
            disorder = torch.randperm(temp_df.shape[0])
            sample_list = disorder[:self.num_shot]
            patientDatas += list(np.array(temp_df.iloc[sample_list, :-1]))
            patientLabels += [self.subtype_number_dict.get(subtype)] * len(sample_list)

            if temp_df.shape[0] >= (self.num_shot * 2):
                not_sampled_list = disorder[self.num_shot:self.num_shot * 2]
            else:
                not_sampled_list = disorder[self.num_shot:]
            queryX += list(np.array(temp_df.iloc[not_sampled_list, :-1]))
            queryY += [self.subtype_number_dict.get(subtype)] * len(not_sampled_list)

        patientDatas = torch.tensor(np.array(patientDatas), dtype=torch.float32)
        patientLabels = torch.tensor(patientLabels)

        queryX = torch.tensor(np.array(queryX), dtype=torch.float32)
        queryY = torch.tensor(queryY)

        return patientDatas, patientLabels, queryX, queryY

    def __len__(self):
        return self.patients_info.shape[0] // self.num_shot // self.num_way
