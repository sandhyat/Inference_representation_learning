import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
import json
from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from pyarrow import feather  # directly writing import pyarrow didn't work
import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# from torch.profiler import profile, record_function, ProfilerActivity

def normalization(data0, mode, normalizing_value, contin_var):
    data = data0.copy()
    if mode == 'mean_std':
        mean = normalizing_value['mean']
        std = normalizing_value['std']
        data[contin_var] = data[contin_var] - mean
        data[contin_var] = data[contin_var] / std
    if mode == 'min_max':
        min_v = normalizing_value['min']
        max_v = normalizing_value['max']
        data[contin_var] = data[contin_var] - min_v
        data[contin_var] = data[contin_var] / max_v
    return data

def preprocess_inference(preops, metadata):
    """

    preops is the input data from EHR with some checks (?)
    metadata is the .json metadata file created during training available in B the box folder
    """

    preops.reset_index(drop=True, inplace=True)
    binary_variables = metadata["binary_var_name"]
    categorical_variables = metadata["categorical_name"]
    ordinal_variables = metadata["ordinal_variables"]
    continuous_variables = metadata["continuous_variables"]
    all_var = binary_variables + categorical_variables + ordinal_variables + continuous_variables

    # this is done because there are some variable which are absent in the wave2 data and hence setting them to nan here so that they can get imputed later
    if len(set(list(all_var)).difference(set(preops.columns))) != 0:  # orlogid_encoded will always be there in the otherway difference
        for i in list(set(list(all_var)).difference(set(preops.columns))):
            preops[i]=np.nan
            if i in categorical_variables: # this needs to be done otherwise the algo doesn't know any levels; ultimately all of them except the number of level examples will be in the nan categrory of the variable
                for j in range(len(metadata['levels'][i])):
                    preops.at[j, i] = metadata['levels'][i][j]


    # encoding the plannedDispo from text to number
    # {"OUTPATIENT": 0, '23 HOUR ADMIT': 1, "FLOOR": 1, "OBS. UNIT": 2, "ICU": 3}
    preops.loc[preops['plannedDispo'] == 'Outpatient', 'plannedDispo'] = 0
    preops.loc[preops['plannedDispo'] == 'Floor', 'plannedDispo'] = 1
    preops.loc[preops['plannedDispo'] == 'Obs. unit', 'plannedDispo'] = 2
    preops.loc[preops['plannedDispo'] == 'ICU', 'plannedDispo'] = 3
    if '' in list(preops['plannedDispo'].unique()):
        preops.loc[preops['plannedDispo'] == '', 'plannedDispo'] = np.nan
    preops['plannedDispo'] = preops['plannedDispo'].astype('float') # needed to convert this to float because the nans were not getting converted to int and this variable is object type



    preops_ohe = preops.copy()[set(binary_variables + categorical_variables + ordinal_variables + continuous_variables)]

    for i in binary_variables:
        preops_ohe[i].fillna(0, inplace=True)

    # this is kind of hardcoded; check your data beforehand for this; fixed this
    # this is done because there were two values for missing token (nan and -inf)
    # NOTE: try the isfinite function defined above
    # this section creates NaNs only to be filled in later. it harmonizes the different kinds of not-a-number representations
    temp_list = [i for i in preops_ohe['PlannedAnesthesia'].unique() if np.isnan(i)] + [i for i in preops_ohe[
        'PlannedAnesthesia'].unique() if math.isinf(i)]
    if temp_list != []:
        preops_ohe['PlannedAnesthesia'].replace(temp_list, np.NaN, inplace=True)

    if 'plannedDispo' in preops_ohe.columns:
        preops_ohe['plannedDispo'].replace('', np.NaN, inplace=True)

    for name in categorical_variables:
        preops_ohe[name] = preops_ohe[name].astype('category')
    for a in preops_ohe.columns:
        if preops_ohe[a].dtype == 'bool':
            preops_ohe[a] = preops_ohe[a].astype('int32')
        if preops_ohe[a].dtype == 'int32':
            if (a in categorical_variables) and (a not in ordinal_variables):
                preops_ohe[a] = pd.Series(
                    pd.Categorical(preops_ohe[a], categories=metadata['levels'][a], ordered=False))

    # one hot encoding
    # this is reverse from how I would have thought about it. It starts with the list of target columns, gets the value associated with that name, then scans for values matching the target
    # i probably would have used pd.get_dummies, concat, drop cols not present in the original, add constant 0 cols that are missing. I think this works as-is
    encoded_var = metadata['encoded_var']
    for ev in encoded_var:
        preops_ohe[ev] = 0
        ev_name = ev.rsplit("_", 1)[0]
        ev_value = ev.rsplit("_", 1)[1]
        if ev_value != 'nan':
            if len(preops[ev_name].unique()) < 2:
                dtype_check = preops[ev_name].unique()[0]
            else:
                dtype_check = preops[ev_name].unique()[1]
            if type(dtype_check) == np.float64 or type(dtype_check) == np.int64:
                preops_ohe[ev] = np.where(preops_ohe[ev_name].astype('float') == float(ev_value), 1, 0)
            elif type(dtype_check) == bool:
                preops_ohe[ev] = np.where(preops[ev_name].astype('str') == ev_value, 1, 0)
            else:
                preops_ohe[ev] = np.where(preops_ohe[ev_name] == ev_value, 1, 0)
    # this for loop checks if the categorical variable doesn't have 1 in any non-NAN value column and then assigns 1 in the nan value column
    # this is done because the type of nans in python are complicated and different columns had different type of nans
    for i in categorical_variables:
        name = str(i) + "_nan"
        lst = [col for col in encoded_var if (i == col.rsplit("_", 1)[0]) and (col != name)]
        preops_ohe['temp_col'] = preops_ohe[lst].sum(axis=1)
        preops_ohe[name] = np.where(preops_ohe['temp_col'] == 1, 0, 1)
        preops_ohe.drop(columns=['temp_col'], inplace=True)
    preops_ohe.drop(columns=categorical_variables, inplace=True)
    # mean imputing and scaling the continuous variables
    preops_ohe[continuous_variables].fillna(
        dict(zip(metadata['norm_value_cont']['cont_names'], metadata["train_mean_cont"])),
        inplace=True)  ## warning about copy
    # this is done because nan that are of float type is not recognised as missing by above commands
    for i in continuous_variables:
        if preops_ohe[i].isna().any() == True:
            preops_ohe[i].replace(preops_ohe[i].unique().min(),
                                  dict(zip(metadata['norm_value_cont']['cont_names'], metadata["train_mean_cont"]))[i],
                                  inplace=True)
    preops_ohe = normalization(preops_ohe, 'mean_std', metadata['norm_value_cont'], continuous_variables)
    # median Imputing_ordinal variables
    # imputing
    for i in ordinal_variables:
        preops_ohe.loc[:, i] = pd.to_numeric(preops_ohe[i], errors='coerce').fillna(
            dict(zip(metadata['norm_value_ord']['ord_names'], metadata["train_median_ord"]))[i])
        # replace(preops_ohe[i].unique().min(), dict(zip(metadata['norm_value_ord']['ord_names'], metadata["train_median_ord"]))[i], inplace=True) # min because nan is treated as min
    # normalization
    preops_ohe = normalization(preops_ohe, 'mean_std', metadata['norm_value_ord'], ordinal_variables)
    preops_ohe = preops_ohe.reindex(metadata["column_all_names"], axis=1)

    if "person_integer" in preops_ohe.columns:
        preops_ohe.rename({"person_integer":"orlogid_encoded"}, axis=1, inplace=True)

    preops_ohe['orlogid_encoded'] = preops['orlogid_encoded']

    return preops_ohe

def load_epic(dataset, modality_to_uselist,data_dir):  # dataset is whether it is flowsheets or meds, list has the name of all modalities that will be used

    # creating modality dictionary
    output_to_return_train = {}
    output_to_return_orlogids = {}
    if 'homemeds' in modality_to_uselist:

        home_meds = pd.read_csv(data_dir + 'home_med_cui.csv', low_memory=False)
        Drg_pretrained_embedings = pd.read_csv(data_dir + 'df_cui_vec_2sourceMappedWODupl.csv')
        home_meds = home_meds.drop_duplicates(subset=['orlogid_encoded','rxcui'])  # because there exist a lot of duplicates if you do not consider the dose column which we dont as of now
        home_meds_embedded = home_meds[['orlogid_encoded', 'rxcui']].merge(Drg_pretrained_embedings, how='left',
                                                                           on='rxcui')
        home_meds_embedded.drop(columns=['code', 'description', 'source'], inplace=True)
        home_meds_embedsum = home_meds_embedded.groupby("orlogid_encoded").sum().reset_index()
        output_to_return_orlogids['homemeds'] = home_meds_embedsum['orlogid_encoded']
        home_meds_embedsum = home_meds_embedsum.drop(columns=["orlogid_encoded",'rxcui'])

        # scaling only the embeded homemed version
        scaler_hm = StandardScaler()
        scaler_hm.fit(home_meds_embedsum)
        train_X_hm = scaler_hm.transform(home_meds_embedsum)

        output_to_return_train['homemeds'] = train_X_hm

        del train_X_hm

    if 'pmh' in modality_to_uselist:

        pmh_emb_sb = pd.read_csv(data_dir + 'pmh_sherbert.csv')
        pmh_embeded = pmh_emb_sb.groupby("orlogid_encoded").sum().reset_index()
        output_to_return_orlogids['pmh'] = pmh_embeded['orlogid_encoded']
        pmh_embeded = pmh_embeded.drop(columns=["orlogid_encoded"])

        # scaling the pmh
        scaler_pmh = StandardScaler()
        scaler_pmh.fit(pmh_embeded)
        train_X_pmh = scaler_pmh.transform(pmh_embeded)

        output_to_return_train['pmh'] = train_X_pmh

        del train_X_pmh

    if 'problist' in modality_to_uselist:

        prob_list_emb_sb = pd.read_csv(data_dir + 'preproblems_sherbert.csv')
        prob_list_embeded = prob_list_emb_sb.groupby("orlogid_encoded").sum().reset_index()
        output_to_return_orlogids['problist'] = prob_list_embeded['orlogid_encoded']
        prob_list_embeded = prob_list_embeded.drop(columns=["orlogid_encoded"])

        # scaling the prob_list
        scaler_problist = StandardScaler()
        scaler_problist.fit(prob_list_embeded)
        train_X_problist = scaler_problist.transform(prob_list_embeded)

        output_to_return_train['problist'] = train_X_problist

        del train_X_problist

    if 'cbow' in modality_to_uselist:
        bow_input = pd.read_csv(data_dir + 'cbow_proc_text.csv')
        bow_cols = [col for col in bow_input.columns if 'BOW' in col]
        bow_input['BOW_NA'] = np.where(np.isnan(bow_input[bow_cols[0]]), 1, 0)
        cbow_final = bow_input.fillna(0)
        output_to_return_orlogids['cbow'] = cbow_final['orlogid_encoded']
        cbow_final = cbow_final.drop(columns=["orlogid_encoded"])

        scaler_cbow = StandardScaler()
        scaler_cbow.fit(cbow_final)
        train_X_cbow = scaler_cbow.transform(cbow_final)

        output_to_return_train['cbow'] = train_X_cbow


        del train_X_cbow

    if ('preops_o' in modality_to_uselist) or ('preops_l' in modality_to_uselist):

        preops = feather.read_feather(data_dir + 'preops_reduced_for_training.feather')
        preops = preops.drop(['MRN_encoded'], axis=1)

        # read the metadata file, seperate out the indices of labs and encoded labs from it and then use them to seperate in the laoded processed preops files above
        md_f1 = open(data_dir + 'preops_metadataicu.json')
        metadata_icu = json.load(md_f1)
        all_column_names = metadata_icu['column_all_names']
        all_column_names.remove('person_integer')

        # processing the data
        preops_train = preprocess_inference(preops, metadata_icu)
        output_to_return_orlogids['preops_o'] = preops_train['orlogid_encoded']
        output_to_return_orlogids['preops_l'] = preops_train['orlogid_encoded']
        preops_train = preops_train.drop(columns=["orlogid_encoded"]).to_numpy()

        # lab names
        f = open(data_dir + 'used_labs.txt')
        preoplabnames = f.read()
        f.close()
        preoplabnames_f = preoplabnames.split('\n')[:-1]

        # labs_to_sep = [i for i in all_column_names: if i in preoplabnames_f elif i.split("_")[:-1] in preoplabnames_f]
        labs_to_sep = []
        for i in all_column_names:
            if i in preoplabnames_f:
                labs_to_sep.append(i)
            else:
                try:
                    if i.split("_")[:-1][0] in preoplabnames_f:
                        labs_to_sep.append(i)
                except IndexError:
                    pass

        lab_indices_Sep = [all_column_names.index(i) for i in labs_to_sep]
        # preop_indices = [i for i in range(len(all_column_names)) if i not in lab_indices_Sep]

        # this is when the pmh and problist modalities are being used
        if 'pmh' in modality_to_uselist or 'problist' in modality_to_uselist:
            # dropping the pmh and problist columns from the preop list
            to_drop_old_pmh_problist = ["MentalHistory_anxiety", "MentalHistory_bipolar", "MentalHistory_depression",
                                        "MentalHistory_schizophrenia", "PNA", "delirium_history", "MentalHistory_adhd",
                                        "MentalHistory_other", "opioids_count", "total_morphine_equivalent_dose",
                                        'pre_aki_status', 'preop_ICU', 'preop_los']

            preop_indices = [all_column_names.index(i) for i in all_column_names if
                             i not in (labs_to_sep + to_drop_old_pmh_problist)]
        else:
            preop_indices = [all_column_names.index(i) for i in all_column_names if i not in (labs_to_sep)]

        preops_train_true = preops_train[:, preop_indices]

        preops_train_labs = preops_train[:, lab_indices_Sep]

        # is the scaling needed again as the preops have been processes already?
        scaler = StandardScaler()
        scaler.fit(preops_train_true)
        train_X_pr_o = scaler.transform(preops_train_true)  # o here means only the non labs

        # is the scaling needed again as the preops have been processes already?
        scaler = StandardScaler()
        scaler.fit(preops_train_labs)
        train_X_pr_l = scaler.transform(preops_train_labs)  # l here means preop labs

        output_to_return_train['preops_o'] = train_X_pr_o
        output_to_return_train['preops_l'] = train_X_pr_l

        del train_X_pr_o, train_X_pr_l

    return output_to_return_train, output_to_return_orlogids


