import torch
import numpy as np
import pandas as pd
import argparse
import os
import sys
# from datetime import datetime
import json
import time
import datetime
from Multiview_CL_modular import MVCL_f_m_sep
import datautils_inf
from utils import init_dl_program
from tasks import scarf_model as preop_model
from torch.utils.data import DataLoader

def save_checkpoint_callback(
        save_every=1,
        unit='epoch'
):
    assert unit in ('epoch', 'iter')

    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')

    return callback


if __name__ == '__main__':

    # presetting the number of threads to be used
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    torch.cuda.set_per_process_memory_fraction(1.0, device=None)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('--preops', action="store_true",
                        help='output preop representation using the trained model')
    parser.add_argument('--cbow', action="store_true",
                        help='output cbow representation using the trained model')
    parser.add_argument('--pmh', action="store_true",
                        help='output pmh representation using the trained model')
    parser.add_argument('--problist', action="store_true",
                        help='output problist representation using the trained model')
    parser.add_argument('--homemeds', action="store_true",
                        help='output homemeds representation using the trained model')
    parser.add_argument('--gpu', type=int, default=0,
                        help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--data_dir', type=str, default=None, help='path to the directory where the raw data is residing')
    parser.add_argument('--model_dir', type=str, default=None, help='path to the directory where the trained models are saved')
    parser.add_argument('--output_dir', type=str, default=None, help='path to the directory where the representation files are saved')
    parser.add_argument('--max-train-length', type=int, default=3000,
                        help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--max-threads', type=int, default=None,
                        help='The maximum allowed number of threads used by this process')
    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    all_modality_list = ['pmh', 'problist', 'homemeds', 'preops_o', 'preops_l', 'cbow']
    repr_modality = []
    if eval('args.preops') == True:
        repr_modality.append('preops_o')
        repr_modality.append('preops_l')

    for i in all_modality_list:
        if i in ['preops_o', 'preops_l']:
            print('for completeness') # should have taken care of in a different way
        elif eval('args.' + str(i)) == True:
            repr_modality.append(i)

    # run_dir = 'training/icu_modal__preops_o_preops_l_cbow_flow_meds_alerts_pmh_problist_homemeds_postopcomp_multipleCL_20240509_135453'  This one also has the ts models and the meta data saved
    run_dir = args.model_dir
    # datadir = '/home/trips/Epic_ContrastiveLearning_all_modalities/Inference_Dir/'
    datadir = args.data_dir

    metadata_file = run_dir + '/model_metadata.json'
    with open(metadata_file) as f:
        config = json.load(f)

    print('Loading data... ', end='')
    proc_modality_dict_train, orlogid_list = datautils_inf.load_epic(args.dataset, repr_modality, data_dir=datadir)

    t = time.time()

    if 'preops_l' in repr_modality:
        proc_modality_dict_train['preops_l'] = torch.tensor(proc_modality_dict_train['preops_l'], dtype=torch.float)

    if 'preops_o' in repr_modality:
        proc_modality_dict_train['preops_o'] = torch.tensor(proc_modality_dict_train['preops_o'], dtype=torch.float)

    if 'cbow' in repr_modality:
        proc_modality_dict_train['cbow'] = torch.tensor(proc_modality_dict_train['cbow'], dtype=torch.float)

    if 'homemeds' in repr_modality:
        proc_modality_dict_train['homemeds'] = torch.tensor(proc_modality_dict_train['homemeds'], dtype=torch.float)

    if 'pmh' in repr_modality:
        proc_modality_dict_train['pmh'] = torch.tensor(proc_modality_dict_train['pmh'], dtype=torch.float)

    if 'problist' in repr_modality:
        proc_modality_dict_train['problist'] = torch.tensor(proc_modality_dict_train['problist'], dtype=torch.float)

    device = init_dl_program(args.gpu, seed=config['seed_used'], max_threads=args.max_threads)
    # breakpoint()
    model = MVCL_f_m_sep(device=device, **config)

    if ('preops_o' in repr_modality) or ('preops_l' in repr_modality):
        train_pr = proc_modality_dict_train['preops_o']
        train_pr_l = proc_modality_dict_train['preops_l']

        train_ds = preop_model.ExampleDataset(train_pr)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
        train_repr_pr = model.pr_dataset_embeddings(train_loader, inf=1)

        train_ds = preop_model.ExampleDataset(train_pr_l)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
        train_repr_pr_l = model.pr_l_dataset_embeddings(train_loader, inf=1)

        train_rep_idx_pr = pd.concat([orlogid_list['preops_o'].reset_index(drop=True), pd.DataFrame(train_repr_pr,columns=['Col' + str(i) for i in range(train_repr_pr.shape[-1])]).reset_index(drop=True)], axis=1)
        train_rep_idx_pr_l = pd.concat([orlogid_list['preops_l'].reset_index(drop=True), pd.DataFrame(train_repr_pr_l,columns=['Col' + str(i) for i in range(train_repr_pr_l.shape[-1])]).reset_index(drop=True)], axis=1)
        train_rep_idx_pr.to_csv(str(args.output_dir) + 'Preop_repres_from_all_modalities.csv', index=False)
        train_rep_idx_pr_l.to_csv(str(args.output_dir) + 'Preop_labs_repres_from_all_modalities.csv', index=False)

    if 'cbow' in repr_modality:
        train_bw = proc_modality_dict_train['cbow']
        train_ds = preop_model.ExampleDataset(train_bw)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
        train_repr_bw = model.cbow_dataset_embeddings(train_loader, inf=1)
        train_rep_idx_bw = pd.concat([orlogid_list['cbow'].reset_index(drop=True), pd.DataFrame(train_repr_bw,columns=['Col' + str(i) for i in range(train_repr_bw.shape[-1])]).reset_index(drop=True)], axis=1)
        train_rep_idx_bw.to_csv(str(args.output_dir) + 'Cbow_repres_from_all_modalities.csv', index=False)


    if 'homemeds' in repr_modality:
        train_hm = proc_modality_dict_train['homemeds']
        train_ds = preop_model.ExampleDataset(train_hm)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
        train_repr_hm = model.hm_dataset_embeddings(train_loader, inf=1)
        train_rep_idx_hm = pd.concat([orlogid_list['homemeds'].reset_index(drop=True), pd.DataFrame(train_repr_hm,columns=['Col' + str(i) for i in range(train_repr_hm.shape[-1])]).reset_index(drop=True)], axis=1)
        train_rep_idx_hm.to_csv(str(args.output_dir) + 'Homemeds_repres_from_all_modalities.csv', index=False)


    if 'pmh' in repr_modality:
        train_pmh = proc_modality_dict_train['pmh']
        train_ds = preop_model.ExampleDataset(train_pmh)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
        train_repr_pmh = model.pmh_dataset_embeddings(train_loader, inf=1)
        train_rep_idx_pmh = pd.concat([orlogid_list['pmh'].reset_index(drop=True), pd.DataFrame(train_repr_pmh,columns=['Col' + str(i) for i in range(train_repr_pmh.shape[-1])]).reset_index(drop=True)], axis=1)
        train_rep_idx_pmh.to_csv(str(args.output_dir) + 'PMH_repres_from_all_modalities.csv', index=False)


    if 'problist' in repr_modality:
        train_pblist = proc_modality_dict_train['problist']
        train_ds = preop_model.ExampleDataset(train_pblist)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
        train_repr_pblist = model.problist_dataset_embeddings(train_loader, inf=1)
        train_rep_idx_pblist = pd.concat([orlogid_list['problist'].reset_index(drop=True), pd.DataFrame(train_repr_pblist,columns=['Col' + str(i) for i in range(train_repr_pblist.shape[-1])]).reset_index(drop=True)], axis=1)
        train_rep_idx_pblist.to_csv(str(args.output_dir) + 'Problist_repres_from_all_modalities.csv', index=False)


    # this would need access to alerts which current datautils_inf file is not capable of
    # however one can modify the following function which will can take the alerts directly and would need the ts saved models in addition to the preops saved models
    # association_metrics_dict = model.associationBTWalertsANDrestmodalities(proc_modality_dict_train)
