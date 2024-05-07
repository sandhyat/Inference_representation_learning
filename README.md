# Inference_representation_learning

The main file to run is inference_modular.py
The code expects that you have trained model which is saved in the training folder, a data directory and an output directory.

For the saved model provided in the training folder, one can use the following command to obtain the representation for all the preops.

```
python inference_modular.py Epic_Dataset --preops --cbow --pmh --problist --homemeds --outcome=icu --medid_embed_dim=17 --repr_dims_f=349 --preops_rep_dim_o=78 --preops_rep_dim_l=96 --cbow_rep_dim=112 --homemeds_rep_dim=217 --pmh_rep_dim=89 --prob_list_rep_dim=50 --outcome_rep_dim=64 --proj_dim=111 --proj_head_depth=5 --weight_preops=0.608 --weight_ts_preops=0.7849 --weight_outcomes=0.7679 --weight_std=0.2834 --weight_cov=0.7875 --weight_mse=0.0 --weight_ts_cross=0.2922 --batch-size=24 --epochs=2 --iters=290 --lr=0.0001 --seed=6936 --output_dir='./' --model_dir='training/icu_modal__preops_o_preops_l_cbow_flow_meds_alerts_pmh_problist_homemeds_postopcomp_multipleCL_20240506_134734' --data_dir='/home/trips/Epic_ContrastiveLearning_all_modalities/Inference_Dir/'
```

If one is interested in only obtaining the representation for selected modality, they can add or remove the following arguments.
``` --preops --cbow --pmh --problist --homemeds ``` 

Note that preops argument will give you preops labs too. outcome argument is irrelevant and is just a placeholder.

The input files or the format that are needed/compatible with are as follows:
1) preops_reduced_for_training.feather
2) cbow_proc_text.csv
3) pmh_sherbert.csv
4) preproblems_sherbert.csv
5) home_med_cui.csv

Additionally, the following files are needed too:
1) used_labs.txt : This is used to seperate the preops labs.
2) df_cui_vec_2sourceMappedWODupl.csv : This is is used to obtain the embeddings for homemeds using a datasource based on [this paper](https://arxiv.org/pdf/1804.01486) and the raw embeddings from [here](https://figshare.com/s/00d69861786cd0156d81).
3) preops_metadataicu.json : This is used for preocessing the preops files.

Currently, all these files for test purposes are available at '/home/trips/Epic_ContrastiveLearning_all_modalities/Inference_Dir/'.

One can run this inside a docker container using the docker121720/pytorch-for-ts:0.5 image.