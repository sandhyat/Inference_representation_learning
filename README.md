# Inference_representation_learning

The main file to run is inference_modular.py
The code expects that you have trained model and the corresponding json file to instantiate the model which is saved in the training folder, a data directory and an output directory.

For the saved model provided in the training folder, one can use the following command to obtain the representation for all the preops.

```
python inference_modular.py Epic_Dataset --preops --cbow --pmh --problist --homemeds --output_dir='./' --model_dir='training/icu_modal__preops_o_preops_l_cbow_flow_meds_alerts_pmh_problist_homemeds_postopcomp_multipleCL_20240509_135453' --data_dir='/home/trips/Epic_ContrastiveLearning_all_modalities/Inference_Dir/'
```

If one is interested in only obtaining the representation for selected modality, they can add or remove the following arguments.
``` --preops --cbow --pmh --problist --homemeds ``` 

Note that preops argument will give you preops labs too.

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