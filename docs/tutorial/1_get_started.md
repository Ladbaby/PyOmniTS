# 🚀 Get Started

This tutorial guides you running existing models, datasets, and loss functions.

## 1. ⏬ Clone the Repository

```shell
cd /path/to/your/project
git clone https://github.com/Ladbaby/PyOmniTS.git
```

## 2. 💿 Prepare the Environment

- Create a new Python virtual environment via the tool of your choice, and activate it. For example, using [Miniconda](https://docs.conda.io/en/latest/miniconda.html)/[Anaconda](https://www.anaconda.com/):

    ```
    conda create -n pyomnits python=3.11
    conda activate pyomnits
    ```
    Python 3.10~3.11 have been tested.

- Install dependencies.

    ```shell
    pip install -r requirements.txt
    ```

    > 🔥Note: some packages are only used by a few models/datasets, which are optional. See comments in `requirements.txt`.

## 3. 💾 Prepare Datasets

### 3.1 Regular

Get them from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) provided by [Time-Series-Library](https://github.com/thuml/Time-Series-Library), which includes the following datasets in this repository:

- ECL (electricity)
- ETTh1 (ETT-small)
- ETTm1 (ETT-small)
- ILI (illness)
- Traffic (traffic)
- Weather (weather)

And place them under `storage/datasets` folder of this project (create the folder if not exists, or you can use symbolic link `ln -s` to redirect to existing dataset files).

You will get the following file structure under `storage/datasets`:

```
.
├── electricity
│   └── electricity.csv
├── ETT-small
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── illness
│   └── national_illness.csv
├── traffic
│   └── traffic.csv
└── weather
    └── weather.csv
```

### 3.2 Irregular

#### 3.2.1 Human Activity

No need to prepare in advance.
Our code will automatically download then preprocess it if you want to train on it.

The following file structure will be found under `storage/datasets`, after the code finish preprocessing:
```
.
└── HumanActivity
    ├── processed
    │   └── data.pt
    └── raw
        └── ConfLongDemo_JSI.txt
```

#### 3.2.2 MIMIC III

Since MIMIC III requires credentialed access:
- Request for raw data from [here](https://physionet.org/content/mimiciii/1.4/).
- Data preprocessing

    Choose one of the options:

    - Option 1: Use the revised scripts in PyOmniTS.
        - Create a new virtual environment with Python 3.7, numpy 1.21.6, and pandas 1.3.5

            ```shell
            conda create -n python37 python=3.7
            conda activate python37
            pip install numpy==1.21.6 pandas==1.3.5
            ```
        - `python data/dependencies/MIMIC_III/preprocess/0_run_all.py`
    - Option 2: Use the original scripts in gru_ode_bayes.
        - Follow the processing scripts in [gru_ode_bayes](https://github.com/edebrouwer/gru_ode_bayes/tree/master/data_preproc/MIMIC) to get `complete_tensor.csv`.
        - Put the result under `~/.tsdm/rawdata/MIMIC_III_DeBrouwer2019/complete_tensor.csv`.

The following file structure will be found under `~/.tsdm`, after the code finish preprocessing:
```
.
├── datasets
│   └── MIMIC_III_DeBrouwer2019
│       ├── metadata.parquet
│       └── timeseries.parquet
└── rawdata
    └── MIMIC_III_DeBrouwer2019
        └── complete_tensor.csv
```


#### 3.2.2 MIMIC IV

Since MIMIC IV requires credentialed access:
- Request for raw data from [here](https://physionet.org/content/mimiciv/1.0/).
- Data preprocessing

    Choose one of the options:

    - Option 1: Use the revised scripts in PyOmniTS.
        - Create a new virtual environment with Python 3.7, numpy 1.21.6, and pandas 1.3.5

            ```shell
            conda create -n python37 python=3.7
            conda activate python37
            pip install numpy==1.21.6 pandas==1.3.5
            ```
        - `python data/dependencies/MIMIC_IV/preprocess/0_run_all.py`
    - Option 2: Use the original scripts in NeuralFlows.
        - Follow the processing scripts in [NeuralFlows](https://github.com/mbilos/neural-flows-experiments/blob/master/nfe/experiments/gru_ode_bayes/data_preproc) to get `full_dataset.csv`.
        - Put the result under `~/.tsdm/rawdata/MIMIC_IV_Bilos2021/full_dataset.csv`.

The following file structure will be found under `~/.tsdm`, after the code finish preprocessing:
```
.
├── datasets
│   └── MIMIC_IV_Bilos2021
│       └── timeseries.parquet
└── rawdata
    └── MIMIC_IV_Bilos2021
        └── full_dataset.csv
```


#### 3.2.3 PhysioNet'12

No need to prepare in advance.
Our code will automatically download then preprocess it if you want to train on it.

The following file structure will be found under `~/.tsdm`, after the code finish preprocessing:
```
.
├── datasets
│   └── Physionet2012
│       ├── Physionet2012-set-A-sparse.tar
│       ├── Physionet2012-set-B-sparse.tar
│       └── Physionet2012-set-C-sparse.tar
└── rawdata
    └── Physionet2012
        ├── set-a.tar.gz
        ├── set-b.tar.gz
        └── set-c.tar.gz
```

#### 3.2.4 USHCN

No need to prepare in advance.
Our code will automatically download then preprocess it if you want to train on it.

The following file structure will be found under `~/.tsdm`, after the code finish preprocessing:
```
.
├── datasets
│   └── USHCN_DeBrouwer2019
│       └── USHCN_DeBrouwer2019.parquet
└── rawdata
    └── USHCN_DeBrouwer2019
        └── small_chunked_sporadic.csv
```

## 4. 🔥 Training

Training scripts are located in `scripts` folder.
For example, to train mTAN on dataset Human Activity:

```shell
sh scripts/mTAN/HumanActivity.sh
```

Training results will be organized in `storage/results/DATASET_NAME/MODEL_NAME/MODEL_ID_TIME`

## 5. ❄️ Testing

Testing will be automatically conducted once the training finished. 
If you wish to run test only, change command line argument `--is_training` in training script from `1` to `0` and run the script.

Testing result `metric.json` will be saved in `storage/results/DATASET_NAME/MODEL_NAME/MODEL_ID_TIME/eval_TIME`
