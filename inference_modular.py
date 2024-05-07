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
    parser.add_argument('--withoutCL', action="store_true",
                        help='does not use CL but instead directly trains XGBT based on the modalities given')
    parser.add_argument('--preops', action="store_true",
                        help='output preop representation using the trained model')
    parser.add_argument('--cbow', action="store_true",
                        help='output cbow representation using the trained model')
    # parser.add_argument('--meds', action="store_true",
    #                     help='Whether to add meds to ts representation in case of epic loader')
    # parser.add_argument('--alerts', action="store_true",
    #                     help='Whether to add alerts to ts representation in case of epic loader')
    parser.add_argument('--pmh', action="store_true",
                        help='output pmh representation using the trained model')
    parser.add_argument('--problist', action="store_true",
                        help='output problist representation using the trained model')
    parser.add_argument('--homemeds', action="store_true",
                        help='output homemeds representation using the trained model')
    # parser.add_argument('--postopcomp', action="store_true",
    #                     help='Whether to add postop complications to ts representation in case of epic loader')
    parser.add_argument('--outcome', type=str, required=True, help='The postoperative outcome of interest')
    # parser.add_argument('--all_rep', action='store_true',
    #                     help='Whether to use the representation of all the modalities of only that of time series (flow and meds); to be used with very rare outcomes such as PE or pulm')
    parser.add_argument('--medid_embed_dim', type=int, default=5,
                        help="Dimension to which medid is embedded to before final representations are learnt using ts2vec.")
    parser.add_argument('--gpu', type=int, default=0,
                        help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr_dims_f', type=int, default=320,
                        help='The representation dimension for flowsheets (defaults to 320)')
    parser.add_argument('--repr_dims_m', type=int, default=None,
                        help='The representation dimension for medications (defaults to 320)')
    parser.add_argument('--repr_dims_a', type=int, default=None,
                        help='The representation dimension for alerts (defaults to 320)')
    parser.add_argument('--preops_rep_dim_o', type=int, default=80,
                        help=' The representation dimension for preops (originally 101 dimension) ')
    parser.add_argument('--preops_rep_dim_l', type=int, default=84,
                        help=' The representation dimension for labs (originally 110 dimension) ')
    parser.add_argument('--cbow_rep_dim', type=int, default=101,
                        help=' The representation dimension for cbow (originally 101 dimension) ')
    parser.add_argument('--outcome_rep_dim', type=int, default=50,
                        help=' The representation dimension for the outcomes view (originallly 57 + mask for 18) ')
    parser.add_argument('--homemeds_rep_dim', type=int, default=256,
                        help=' The representation dimension for the homemeds view (currently 500) ')
    parser.add_argument('--pmh_rep_dim', type=int, default=85,
                        help=' The representation dimension for the pmh view (currently 1024, 123 for sherbet version) ')
    parser.add_argument('--prob_list_rep_dim', type=int, default=85,
                        help=' The representation dimension for the problem list view (currently 1024, 123 for sherbet version) ')
    parser.add_argument('--proj_dim', type=int, default=100,
                        help=' Common dimension where all the views are projected to.')
    parser.add_argument('--proj_head_depth', type=int, default=2,
                        help=' Depth of the projection head. Same across all the modalities.')
    parser.add_argument('--weight_preops', type=float, default=0.4, help=' Weight multiplier for the preop loss')
    parser.add_argument('--weight_ts_preops', type=float, default=0.2, help=' Weight multipler for the inter view loss')
    parser.add_argument('--weight_outcomes', type=float, default=0.3, help=' Weight multiplier for the outcome loss')
    parser.add_argument('--weight_std', type=float, default=0.3, help=' Weight multiplier for the std reg term')
    parser.add_argument('--weight_cov', type=float, default=0.3, help=' Weight multiplier for the covariance reg term')
    parser.add_argument('--weight_mse', type=float, default=0,
                        help='Weight multiplier for the between modality mse loss')
    parser.add_argument('--weight_ts_cross', type=float, default=0.3,
                        help='Weight multiplier for the between time series modality')
    parser.add_argument('--data_dir', type=str, default=None, help='path to the directory where the raw data is residing')
    parser.add_argument('--model_dir', type=str, default=None, help='path to the directory where the trained models are saved')
    parser.add_argument('--output_dir', type=str, default=None, help='path to the directory where the representation files are saved')
    parser.add_argument('--max-train-length', type=int, default=3000,
                        help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None,
                        help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed_used', type=int, default=None, help='The random seed that was used while training as it will be in the training folders name')
    parser.add_argument('--max-threads', type=int, default=None,
                        help='The maximum allowed number of threads used by this process')
    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    if (args.withoutCL == True) and (args.all_rep == True):
        print("Incompatible combination")
        exit()

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

    # enforcing representation size choices across the encoders
    if args.repr_dims_m == None or args.repr_dims_a == None:
        args.repr_dims_m = args.repr_dims_f
        args.repr_dims_a = args.repr_dims_f

    # run_dir = 'training/icu_modal__preops_o_preops_l_cbow_flow_meds_alerts_pmh_problist_homemeds_postopcomp_multipleCL_20240506_134734'
    run_dir = args.model_dir
    # datadir = '/home/trips/Epic_ContrastiveLearning_all_modalities/Inference_Dir/'
    datadir = args.data_dir

    print('Loading data... ', end='')
    proc_modality_dict_train, orlogid_list = datautils_inf.load_epic(args.dataset, args.outcome, repr_modality, data_dir=datadir)

    config = dict(
        medid_embed_dim=args.medid_embed_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims_f=args.repr_dims_f,
        output_dims_m=args.repr_dims_m,
        output_dims_a=args.repr_dims_a,
        preops_output_dims_o=args.preops_rep_dim_o,
        preops_output_dims_l=args.preops_rep_dim_l,
        cbow_output_dims=args.cbow_rep_dim,
        homemeds_rep_dims=args.homemeds_rep_dim,
        pmh_rep_dims=args.pmh_rep_dim,
        prob_list_rep_dims=args.prob_list_rep_dim,
        outcome_rep_dims=args.outcome_rep_dim,
        max_train_length=args.max_train_length,
        w_pr=args.weight_preops,
        w_ts_pr=args.weight_ts_preops,
        w_out=args.weight_outcomes,
        w_std=args.weight_std,
        w_cov=args.weight_cov,
        w_mse=args.weight_mse,
        w_ts_cross=args.weight_ts_cross,
        proj_dim=args.proj_dim,
        head_depth=args.proj_head_depth,
        save_dir=run_dir
    )

    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    t = time.time()

    if 'preops_l' in repr_modality:
        proc_modality_dict_train['preops_l'] = torch.tensor(proc_modality_dict_train['preops_l'], dtype=torch.float)
        config['preops_input_dims_l'] = proc_modality_dict_train['preops_l'].shape[1]

    if 'preops_o' in repr_modality:
        proc_modality_dict_train['preops_o'] = torch.tensor(proc_modality_dict_train['preops_o'], dtype=torch.float)
        config['preops_input_dims_o'] = proc_modality_dict_train['preops_o'].shape[1]

    if 'cbow' in repr_modality:
        proc_modality_dict_train['cbow'] = torch.tensor(proc_modality_dict_train['cbow'], dtype=torch.float)
        config['cbow_input_dims'] = proc_modality_dict_train['cbow'].shape[1]

    if 'homemeds' in repr_modality:
        proc_modality_dict_train['homemeds'] = torch.tensor(proc_modality_dict_train['homemeds'], dtype=torch.float)
        config['hm_input_dims'] = proc_modality_dict_train['homemeds'].shape[1]

    if 'pmh' in repr_modality:
        proc_modality_dict_train['pmh'] = torch.tensor(proc_modality_dict_train['pmh'], dtype=torch.float)
        config['pmh_input_dims'] = proc_modality_dict_train['pmh'].shape[1]

    if 'problist' in repr_modality:
        proc_modality_dict_train['problist'] = torch.tensor(proc_modality_dict_train['problist'], dtype=torch.float)
        config['prob_list_input_dims'] = proc_modality_dict_train['problist'].shape[1]

    device = init_dl_program(args.gpu, seed=args.seed_used, max_threads=args.max_threads)

    model = MVCL_f_m_sep(
                device=device,
                seed_used=args.seed_used,
                **config
            )

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


