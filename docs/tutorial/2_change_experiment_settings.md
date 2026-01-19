# ⚙️ Change Experiment Settings

This tutorial guides you changing the settings of your experiments.

## 1. Overview

All available setting options are defined in `utils/configs.py`, where PyOmniTS uses Python's built-in `argparse` package to configure all the settings.

## 2. Change Settings

It is recommended to overwrite the values of these settings via the scripts under the `scripts/` folder.
Changing default values of arguments in `utils/configs.py` is not recommended, which may affect other PyOmniTS components.

## 3. Commonly Used Settings

We take the content of `scripts/mTAN/HumanActivity.sh` as an example:

```shell
use_multi_gpu=0
if [ $use_multi_gpu -eq 0 ]; then
    launch_command="python"
else
    launch_command="accelerate launch"
fi

. "$(dirname "$(readlink -f "$0")")/../globals.sh" # Import shared information from scripts/globals.sh

dataset_name=$(basename "$0" .sh) # file name
dataset_subset_name=""
dataset_id=$dataset_name
get_dataset_info "$dataset_name" "$dataset_subset_name" # Get dataset information from scripts/globals.sh

model_name="$(basename "$(dirname "$(readlink -f "$0")")")" # folder name
model_id=$model_name

seq_len=3000
for pred_len in 300; do
    $launch_command main.py \
        --is_training 1 \
        --collate_fn "collate_fn" \
        --loss "ModelProvidedLoss" \
        --use_multi_gpu $use_multi_gpu \
        --dataset_root_path $dataset_root_path \
        --model_id $model_id \
        --model_name $model_name \
        --dataset_name $dataset_name \
        --dataset_id $dataset_id \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in $n_variables \
        --dec_in $n_variables \
        --c_out $n_variables \
        --train_epochs 300 \
        --patience 10 \
        --val_interval 1 \
        --itr 5 \
        --batch_size 32 \
        --learning_rate 1e-3
done
```

These fields are vital:

- `use_multi_gpu=0`: Change to `1` if you want to enable parallel training via [accelerate](https://huggingface.co/docs/accelerate/en/index).
- `dataset_name=$(basename "$0" .sh)`: Retrieve the file name of current script file, no need to change. After passing to `--dataset_name`, PyOmniTS will automatically find the corresponding dataset class with the exact same file name under `data/data_provider/datasets/`.
- `dataset_id`: Only affects the folder name for storing experiment results, unlike `dataset_name`.
- `model_name="$(basename "$(dirname "$(readlink -f "$0")")")"`: Retrieve the folder name where current script file is placed, no need to change. After passing to `--model_name`, PyOmniTS will automatically find the corresponding model class with the exact same file name under `models/`.
- `model_id`: Only affects the folder name for storing experiment results, unlike `model_name`.
- `seq_len=3000`: The lookback window length of input time series.
- `for pred_len in 300; do`: The forecast window length of forecast targets.
- `--is_training 1`: Change to `0` if you want testing only, instead of training+testing when set to `1`.
- `--collate_fn "collate_fn"`: Some datasets and models need special collate_fn for `torch.utils.data.DataLoader`. e.g., In the above example, PyOmniTS will try to load the function with the exact same name in `data/data_provider/datasets/HumanActivity.py`.
- `--loss "ModelProvidedLoss"`: Loss function for training. PyOmniTS will automatically find the corresponding loss function class with the exact same file name under `loss_fns/`.
- `--train_epochs 300`: Maximum training epochs. Normally, this is never reached, since early stopping is used.
- `--patience 10`: Early stop patience (counter +1 when validation loss is not decreasing).
- `--val_interval 1`: Frequency (epoch) for calculating validation loss. It should be noted that it will affect early stopping (e.g., `--val_interval 2` and `--patience 10` will wait for 20 epoches before early stopping).
- `--itr 5`: Number of runs (for mean/std calculation of metrics).
- `--batch_size 32`: Batch size.
- `--learning_rate 1e-3` Learning rate.

## 4. Other Settings

All settings are available in `utils/configs.py` with detailed comments.

