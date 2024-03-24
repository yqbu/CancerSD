import numpy as np
import pandas as pd
import os


def ignore_file(ignore_item_list, current_file_list):
    for ignore in ignore_item_list:
        if current_file_list.find(ignore) >= 0:
            return True

    return False


def get_omics_data(omics_data_list):
    omics_data = pd.concat(omics_data_list, axis=1)

    omics_duplicated = omics_data.columns.duplicated()
    omics_duplicated = (omics_duplicated == False)
    omics_data = omics_data.T.loc[omics_duplicated].T
    omics_data.replace('NA', np.NaN, inplace=True)
    omics_data.replace('0', np.NaN, inplace=True)
    omics_data.fillna(0, inplace=True)

    return omics_data


def standardize_omics_data(omics_data):
    columns = omics_data.drop(index=omics_data.index).columns.values
    indices = omics_data.drop(columns=omics_data.columns).index.values
    rename_columns = dict(zip(range(omics_data.shape[1]), columns))
    rename_indexes = dict(zip(range(omics_data.shape[0]), indices))

    temp = omics_data.values.astype(float)
    temp[np.isnan(temp)] = 0.0
    temp = np.log2(temp + 1)
    result = pd.DataFrame(temp)
    result.fillna(0, inplace=True)
    result.rename(index=rename_indexes, columns=rename_columns, inplace=True)

    return result


def normalize_omics_data(cancer, omics_name, omics_data):
    columns = omics_data.drop(index=omics_data.index).columns.values
    indices = omics_data.drop(columns=omics_data.columns).index.values
    rename_columns = dict(zip(range(omics_data.shape[1]), columns))
    rename_indexes = dict(zip(range(omics_data.shape[0]), indices))

    x = omics_data.values.astype(float)
    x_min, x_max = x.min(axis=1), x.max(axis=1)
    x_min = x_min[:, np.newaxis]
    x_max = x_max[:, np.newaxis]
    x_norm = (x - x_min) / (x_max - x_min)
    max_min = pd.DataFrame(data=np.concatenate([x_max, x_min], axis=1),
                           index=omics_data.index, columns=['max', 'min'])
    max_min.index.rename(omics_name, inplace=True)
    max_min.to_csv(f'../Data/{cancer}/{omics_name}_max_min.csv')

    result = pd.DataFrame(x_norm)
    result.fillna(0, inplace=True)
    result.rename(index=rename_indexes, columns=rename_columns, inplace=True)

    return result


def get_patients_with_complete_omics(omics_data_dict):
    patientSelected = set()
    for omics_data in omics_data_dict.values():
        if len(patientSelected) == 0:
            patientSelected = set(omics_data.columns.values)
        else:
            patientSelected = set(omics_data.columns.values) & patientSelected

    return list(patientSelected)


def get_patients_diagnoses(path, race=None):
    patient_infos = pd.read_table(path)
    reserved_columns = ['Patient ID', 'Molecular Subtype', 'TNM Stage', 'race']
    patient_diagnoses = patient_infos[reserved_columns]
    patient_diagnoses.set_index('Patient ID', inplace=True)
    if race is not None:
        patient_diagnoses = patient_diagnoses[patient_diagnoses['race'] == race]
    patient_diagnoses = patient_diagnoses.rename(columns={'Molecular Subtype': 'subtype', 'TNM Stage': 'stage'}).T
    patient_diagnoses.index.rename('aspect', inplace=True)
    patient_diagnoses.fillna('unclassifiable', inplace=True)

    return patient_diagnoses


def select_molecules_by_std(df, threshold):
    reserved_molecules = []
    var_list = []
    molecules_with_std = {}
    for row, data in df.iterrows():
        var = data.var()
        var_list.append(var)
        molecules_with_std[row] = var
        if var >= threshold:
            reserved_molecules.append(row)

    return reserved_molecules


def process_diagnoses(cancer, columns, path=None):
    if path is not None:
        diagnoses = pd.read_csv(path).set_index('aspect')
        return diagnoses
    savePath = '../Data/' + cancer + '/patient_diagnose.csv'
    if os.path.exists(savePath):
        diagnoses = pd.read_csv(savePath).set_index('aspect')
        return diagnoses

    root = '/home/laboratory/' + cancer
    newColumns = ['patient', 'subtype', 'stage', 'race']
    diagnoses = pd.read_table(os.path.join(root, 'clinical.tsv'))
    diagnoses = diagnoses.loc[:, columns]
    columnRename = dict(zip(columns, newColumns))
    diagnoses.rename(columns=columnRename, inplace=True)
    diagnoses.set_index('patient', inplace=True)
    diagnoses.dropna(axis=0, inplace=True)
    diagnoses = diagnoses.T
    diagnoses.index.rename('aspect', inplace=True)
    diagnoses.to_csv(savePath, encoding='utf-8')

    return diagnoses


def filter_molecules(omics_name, df, selected_patients, selected_molecules=None):
    current_patients = df.columns
    reserved_patients = list(set(current_patients) & set(selected_patients))
    df = df[reserved_patients]

    current_molecules = df.index
    if selected_molecules is not None:
        missing_molecules = list(set(selected_molecules) - set(current_molecules))
        missing_molecules = pd.DataFrame(data=None, index=missing_molecules, columns=reserved_patients).fillna(0.0)
        reserved_molecules = list(set(current_molecules) & set(selected_molecules))
        df = df.loc[reserved_molecules, :]
        df = pd.concat([df, missing_molecules], axis=0)
        df = df.loc[selected_molecules, :]

    patients = df.columns.tolist()
    df = df.fillna(0)
    df[patients] = df[patients].apply(pd.to_numeric)
    df.index.rename(omics_name, inplace=True)

    return df


def omics_extend_as(df, target):
    missing = list(set(target) - set(df.index.values))
    if len(missing) > 0:
        extension = pd.DataFrame(data=None, columns=df.columns, index=missing)
        extension.fillna(0.0, inplace=True)
        temp = pd.concat([df, extension], axis=0)

        return temp

    return df