import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import process_utils

if __name__ == '__main__':
    cancer = 'STAD'
    columns = ['Patient ID', 'Subtype', 'Neoplasm Disease Stage American Joint Committee on Cancer Code',
               'Race Category']
    diagnoses = process_utils.process_diagnoses(cancer, columns)

    patient_list = diagnoses.columns.values.tolist()
    omics_types = ['methylation', 'miRNA', 'mRNA']
    selected_molecules_dict = {}
    for moleType in omics_types:
        molecules = list(np.load('../../Data/Molecule_selected/' + moleType + '_selected.npy', allow_pickle=True))

        # molecules = []
        # with open('../../Data/'+moleType+'_selected_original.txt', 'r', encoding='utf-8') as txt:
        #     for line in txt:
        #         data_line = line.strip('\n')
        #         if moleType == 'miRNA':
        #             data_line = data_line.strip('-3p')
        #             data_line = data_line.strip('-5p')
        #             data_line = data_line.lower()
        #         if moleType == 'mRNA':
        #             data_line = data_line.split('|')[0]
        #         molecules.append(data_line)
        print(f'{moleType} select: {molecules}')
        print(f'length: {len(molecules)}')
        selected_molecules_dict[moleType] = molecules

    suffix = None
    data_root = '/home/laboratory/'
    data_root = data_root + cancer

    methylation_files_mapping = pd.read_table(os.path.join(data_root, 'methylation_to_patient.tsv'))
    miRNA_files_mapping = pd.read_table(os.path.join(data_root, 'miRNA_to_patient.tsv'))
    mRNA_files_mapping = pd.read_table(os.path.join(data_root, 'mRNA_to_patient.tsv'))

    methylation_files_mapping = methylation_files_mapping.set_index('File ID')['Case ID'].to_dict()
    miRNA_files_mapping = miRNA_files_mapping.set_index('File ID')['Case ID'].to_dict()
    mRNA_files_mapping = mRNA_files_mapping.set_index('File ID')['Case ID'].to_dict()

    omics_data = {'methylation': {}, 'miRNA': {}, 'mRNA': {}}
    ignore_items = ['isoforms', 'annotations', 'parcel', 'partial']
    for omic in omics_data:
        original_omics_path = f'../../Data/{cancer}/{omic}_original.csv'
        if os.path.exists(original_omics_path):
            omics_data[omic] = pd.read_csv(original_omics_path).set_index(omic)
            if 'Unnamed: 0' in omics_data[omic].columns.values:
                omics_data[omic].drop(columns=['Unnamed: 0'], inplace=True)
            continue

        print(f'current processing omics type: {omic}')
        path = os.path.join(data_root, omic)
        folder_list = os.listdir(path)

        data = []
        for folder in tqdm(folder_list):
            os_walk_paths = os.path.join(path, folder)
            for _, _, files in os.walk(os_walk_paths, topdown=False):
                for fileName in files:
                    if process_utils.ignore_file(ignore_items, fileName):
                        continue
                    if omic == 'methylation':
                        with open(os.path.join(os_walk_paths, fileName)) as file:
                            content = file.readlines()
                        methylations = []
                        for line in content:
                            methylations.append(line.replace('\n', '').split('\t'))
                        file.close()

                        methylations = np.asarray(methylations)
                        patientOmicData = pd.DataFrame(methylations, columns=['methylation', 'value'])
                        patientOmicData.set_index('methylation', inplace=True)
                        patientOmicData.rename(columns={'value': methylation_files_mapping[folder]}, inplace=True)
                        data.append(patientOmicData)

                    elif omic == 'miRNA':
                        with open(os.path.join(os_walk_paths, fileName)) as file:
                            content = file.readlines()
                        miRNAs = []
                        for line in content[1:]:
                            lineContent = line.replace('\n', '').split('\t')
                            miRNAs.append([lineContent[0], lineContent[2]])
                        file.close()
                        patientOmicData = pd.DataFrame.from_dict(dict(miRNAs), orient='index')
                        patientOmicData.rename(columns={0: miRNA_files_mapping[folder]}, inplace=True)
                        data.append(patientOmicData)

                    elif omic == 'mRNA':
                        content = pd.read_table(os.path.join(os_walk_paths, fileName), skiprows=1)
                        content.drop(content.head(4).index, inplace=True)
                        content = content[content['gene_type'] == 'protein_coding']
                        content = content[['gene_name', 'fpkm_unstranded']]
                        content.drop_duplicates('gene_name', inplace=True)
                        content.set_index('gene_name', inplace=True)
                        content.rename(columns={'fpkm_unstranded': mRNA_files_mapping[folder]}, inplace=True)
                        data.append(content)

        omics_data[omic] = process_utils.get_omics_data(data)

    for k, v in omics_data.items():
        omics_data[k].index.rename(k, inplace=True)
        omics_data[k].to_csv(f'../../Data/{cancer}/{k}_original.csv', encoding='utf-8')
        print(f'moleType: {k}')
        print(v)

    omics_data['miRNA'] = process_utils.standardize_omics_data(omics_data['miRNA'])
    omics_data['mRNA'] = process_utils.standardize_omics_data(omics_data['mRNA'])

    patients_with_incompleteMe = [patient for patient in patient_list if
                                  patient not in omics_data['methylation'].columns]
    patients_with_incompleteMi = [patient for patient in patient_list if patient not in omics_data['miRNA'].columns]
    patients_with_incompleteM = [patient for patient in patient_list if patient not in omics_data['mRNA'].columns]

    np.save('../../Data/' + cancer + '/patient_lack_methylation.npy', np.array(patients_with_incompleteMe))
    np.save('../../Data/' + cancer + '/patient_lack_miRNA', np.array(patients_with_incompleteMi))
    np.save('../../Data/' + cancer + '/patient_lack_mRNA', np.array(patients_with_incompleteM))

    print('omics filtering...')
    methylation_df = process_utils.filter_molecules('methylation', omics_data['methylation'], patient_list)
    miRNA_df = process_utils.filter_molecules('miRNA', omics_data['miRNA'], patient_list)
    mRNA_df = process_utils.filter_molecules('mRNA', omics_data['mRNA'], patient_list)

    print(methylation_df)
    print(miRNA_df)
    print(mRNA_df)

    methylation_reserved = list(set(omics_data['methylation'].columns) & set(patient_list))
    methylation_df = omics_data['methylation'][methylation_reserved]
    methylation_df = methylation_df.loc[selected_molecules_dict['methylation'], :]
    methylation_df.index.rename('methylation', inplace=True)
    columns = methylation_df.columns.tolist()
    methylation_df[columns] = methylation_df[columns].apply(pd.to_numeric)

    miRNA_reserved = list(set(omics_data['miRNA'].columns) & set(patient_list))
    miRNA_df = omics_data['miRNA'][miRNA_reserved]
    miRNA_df = miRNA_df.loc[selected_molecules_dict['miRNA'], :]
    miRNA_df.index.rename('miRNA', inplace=True)

    mRNA_reserved = list(set(omics_data['mRNA'].columns) & set(patient_list))
    mRNA_df = omics_data['mRNA'][mRNA_reserved]
    mRNA_df = mRNA_df.loc[selected_molecules_dict['mRNA'], :]
    mRNA_df.index.rename('mRNA', inplace=True)

    print('calculating omics variance...')
    methylation_diff = process_utils.select_molecules_by_std(methylation_df, 0.2)
    miRNA_diff = process_utils.select_molecules_by_std(miRNA_df, 0.1)
    mRNA_diff = process_utils.select_molecules_by_std(mRNA_df, 0.8)

    selected_methylation = list((set(methylation_df.index.values) & set(selected_molecules_dict['methylation']))
                                | set(methylation_diff))
    print(f'the number of selected methylation: {len(selected_methylation)}')
    selected_miRNA = list((set(miRNA_df.index.values) & set(selected_molecules_dict['miRNA'])) | set(miRNA_diff))
    print(f'the number of selected miRNA: {len(selected_miRNA)}')
    selected_mRNA = list((set(mRNA_df.index.values) & set(selected_molecules_dict['mRNA'])) | set(mRNA_diff))
    print(f'the number of selected mRNA: {len(selected_mRNA)}')

    selected_methylation = list(set(methylation_df.index.values) & set(selected_molecules_dict['methylation']))
    selected_miRNA = list(set(miRNA_df.index.values) & set(selected_molecules_dict['miRNA']))
    selected_mRNA = list(set(mRNA_df.index.values) & set(selected_molecules_dict['mRNA']))

    np.save('../../Data/methylation_selected.npy', selected_methylation)
    np.save('../../Data/miRNA_selected.npy', selected_miRNA)
    np.save('../../Data/mRNA_selected.npy', selected_mRNA)

    methylation_df = methylation_df.loc[selected_methylation, :]
    miRNA_df = miRNA_df.loc[selected_miRNA, :]
    mRNA_df = mRNA_df.loc[selected_mRNA, :]

    miRNA_df = process_utils.normalize_omics_data(cancer, 'miRNA', miRNA_df)
    mRNA_df = process_utils.normalize_omics_data(cancer, 'mRNA', mRNA_df)

    methylation_df = process_utils.omics_extend_as(methylation_df, selected_molecules_dict['methylation'])
    methylation_df.index.rename('methylation', inplace=True)
    miRNA_df = process_utils.omics_extend_as(miRNA_df, selected_molecules_dict['miRNA'])
    miRNA_df.index.rename('miRNA', inplace=True)
    mRNA_df = process_utils.omics_extend_as(mRNA_df, selected_molecules_dict['mRNA'])
    mRNA_df.index.rename('mRNA', inplace=True)
    if suffix is not None:
        methylation_df.to_csv('../../Data/' + cancer + '/methylation' + suffix + '.csv', encoding='utf-8')
        miRNA_df.to_csv('../../Data/' + cancer + '/miRNA' + suffix + '.csv', encoding='utf-8')
        mRNA_df.to_csv('../../Data/' + cancer + '/mRNA' + suffix + '.csv', encoding='utf-8')
    print(methylation_df)
    print(miRNA_df)
    print(mRNA_df)
    diagnose = process_utils.get_patients_diagnoses('/home/laboratory/' + cancer + '/clinical.tsv')
    diagnose.to_csv('../../Data/' + cancer + '/patient_diagnose.csv', encoding='utf-8')
