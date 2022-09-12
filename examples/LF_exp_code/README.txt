The structure of the folders is as follows

./PINA/LF-experiment/
|---- dataset/
|----    |---- ${dataset_name}/
|----    |----   |---- X_bow.trn.npz         # Default BoW feature for training split.
|----    |----   |---- X_bow.tst.npz         # Default BoW feature for test split.
|----    |----   |---- Y_bow.npz             # Default BoW feature for labels.
|----    |----   |---- trn_X_Y.txt           # Necessary file for reciprocal pair removal.
|----    |----   |---- tst_X_Y.txt           # Necessary file for reciprocal pair removal.
|----    |----   |---- raw/                  # This folder should contain the downloaded "Raw text" files from the XC Repo [1].
|----    |----   |---- normalized/           # This folder will contain the processed raw text in the "raw" folder.
|---- models                                 # Our trained models will be saved here.
|---- scripts/
|        |---- params/
|        |       |---- xtransformer/         # This folder contains config files (hyperparameters) for training XR-Transformer.
|        |       |---- tfidf/                # This folder contains config files (hyperparameters) for constructing tfidf features.
|    
|        |---- vectorize.sh                  # This is the shell script for constructing tfidf feature. Please double check the path defined therein.
|        |---- xrtransformer.sh              # This is the shell script for training XR-Transformer (and also prediction). Please double check the path defined therein.
|        |---- xrtransformer-XYstack.sh
|        |---- xrtransformer-XYstack-DS.sh
|---- README.txt                             # This file.
|---- DataPrep_forXCrepo.py
|---- pecos_dataform_full.py
|---- PINA_augmentation.py
|---- PrepareXYstack.py
|---- evaluate.py



Tested enviroment and required packages:
python==3.9, libpecos==0.3.0, pyxclib, tqdm, pandas, argparse, json, gdown

- pyxclib github: https://github.com/kunaldahiya/pyxclib
- The rest packages can be installed with pip or automatically included in libpecos.

### Setup enviroment memo:

Setup enviroment:

1. source activate pytorch

2. conda create -n "YOURNAME" python=3.9

3. conda activate YOURNAME

4. pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# CHECK: python -c "import torch; print('torch={}, cuda={}'.format(torch.__version__, torch.cuda.is_available()))"

5. python3 -m pip install libpecos

6. conda install -c conda-forge jupyterlab

7. jupyter lab --no-browser --port=5678

8. conda install -c anaconda ipykernel

9. python -m ipykernel install --user --name=YOURNAME

10. pip3 install gdown

Optional:

1. sudo yum search htop

2. sudo yum install htop


### Before training:

1. Download raw text data from [1] and save it at `./dataset/xmc_data/raw/`. `gunzip` or 'unzip' it.

2. Download BoW features from [1] and save it at `./dataset/xmc_data/`. 'unzip' it if necessary.

3. Run `python3 DataPrep_forXCrepo.py --dataset ${dataset}` for extracting BoW features, labels etc.

4. Run `python3 pecos_dataform_full.py ${dataset}` for text preprocessing.

5. (Optional) If there's no `trn_X_Y.txt` and related files, please further run
```
dataset_name={YOUR DATASET NAME} 
perl convert_format.pl ./dataset/${dataset_name}/train.txt ./dataset/${dataset_name}/trn_X_Xf.txt ./dataset/${dataset_name}/trn_X_Y.txt
perl convert_format.pl ./dataset/${dataset_name}/test.txt ./dataset/${dataset_name}/tst_X_Xf.txt ./dataset/${dataset_name}/tst_X_Y.txt
```

You may also try to run `./scripts/Download_data.sh ${dataset}`. However, it will only works properly for non-titles dataset. For titles dataset, please follow the instruction above.


### To train XR-Transformer from scratch:

0. Run `chmod a+x ./scripts/*` if you cannot execute shell script by directly.

1. Run `./scripts/xtransformer-raw.sh ${model_name} ${feature_name} ${dataset}`. The results will be saved under `./models_LF`.
```
${model_name}: This corresponds to the name of the json file in ./params/xtransformer/
${feature_name}: Choose between "BoW" or else. "BoW" refers to using default BoW features and else of self-built tfidf feature.
${dataset}: The name of dataset.
```

If you want to get the ensembled results directly, run `./scripts/Run_XRT-ensemble.sh ${dataset}`.

### To apply PINA:

If you want to get the ensembled results directly, run `./scripts/Run_PINA+XRT-ensemble.sh ${dataset}`.

Below are the step by step instruction.

0. Run `chmod a+x /scripts/*` if you cannot execute shell script by directly.

2. Run `python3 PrepareXYstack-raw.py --dataset ${dataset}` to prepare pretraining XMC data.

3. Run `./scripts/xtransformer-XYstack-raw.sh ${model_name} ${feature_name} ${dataset}`. This will pretrain a neighborh predictor and extract neighbors for downstream instances.

```
${model_name}: This corresponds to the name of the json file in ./params/xtransformer/
${feature_name}: Choose between "BoW" or else. "BoW" refers to using default BoW features and else of self-built tfidf feature.
${dataset}: The name of dataset.
```

4. `cd ..`. Run `python3 PINA_augmentation.py --model_name ${model_name} --feature_name ${feature_name} --dataset ${dataset} --L_option ${L_option} --Pk ${Pk}`. This will return the PINA augmented instance features for the downstream task (saved under pretraining model folder).

```
${model_name}: This corresponds to the name of the json file in ./params/xtransformer/
${feature_name}: Choose between "BoW" or else. "BoW" refers to using default BoW features and else of self-built tfidf feature.
${dataset}: The name of dataset.
${L_option}: The design of feature for pretraining XMC output space.
${Pk}: #Neighbors to aggregate.
```

### Train one XR-Transformer model with PINA augmentation

0. Complete PINA pretraining described as above.

1. Run `/scripts/xtransformer-XYstack-DS-raw.sh ${model_name} ${DS_model_name} ${feature_name} ${dataset} ${Pk} ${L_option}`. 

```
${model_name}: This is for pretraining model configs.
${DS_model_name}: This is for downstream model configs.
${feature_name}: Choose between "BoW" or else. "BoW" refers to using default BoW features and else of self-built tfidf feature.
${dataset}: The name of dataset.
${L_option}: The design of feature for pretraining XMC output space.
${Pk}: #Neighbors to aggregate.
```

### Obtain the ensemble result

0. (Optional) For each json file under `scripts/params/xtransformer/`, `vM.json` differ with `vM-sN.json` only on random seed. One can either use our prepared json files therein or create your own `vM-sN.json` files for ensemble. N should be 1, 2, 3 ...

1. Run `./scripts/xtransformer-XYstack-DS-raw.sh ${model_name} ${DS_model_name} ${feature_name} ${dataset} ${Pk} ${L_option}` multiple times, where ${DS_model_name} should be different for ensemble (i.e. ${DS_model_name} = `v0-raw,v0-raw-s1,v0-raw-s2`).

2. Run 
```
python3 Ensemble-PINA.py --dataset ${dataset} \
                    --model_name ${model_name} \
                    --DS_model_names ${DS_model_names} \
                    --feature_name ${feature_name} \
                    --ens_name ${ens_name} \
                    --Pk ${Pk} \
                    --L_option ${L_option}
```

Note that `${DS_model_names}` is a string specifying all ensemble model name, seperate with `,`. For example: 'v0-raw,v0-raw-s1,v0-raw-s2'.

3. Run `./scripts/Ensemble_evaluation.sh ${ens_name} ${dataset} ${model_dir}`. Note that `${model_dir}` will be printed out after executing step 2. Just copy paste it here.

### References:

[1] XC Repo: http://manikvarma.org/downloads/XC/XMLRepository.html