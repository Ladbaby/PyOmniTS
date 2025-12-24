# ⚙️ Change Experiment Settings

This tutorial guides you changing the settings of your experiments.

## 1. Overview

All available setting options are defined in `utils/configs.py`, where PyOmniTS uses Python's built-in `argparse` package to configure all the settings.

## 2. Change Settings

It is recommended to overwrite the values of these settings via the scripts under `scripts` folder.

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

These fields are commonly changed during experiments:

- `use_multi_gpu=0`: change to `1` if you want to enable parallel training via [accelerate](https://huggingface.co/docs/accelerate/en/index).
- `seq_len=3000`: the lookback window length of input time series
- `for pred_len in 300; do`: the forecast window length of forecast targets
- `--is_training 1`: change to `0` if you want testing only instead of training+testing.
- `--train_epochs 300`: maximum training epochs
- `--patience 10`: early stop patience (epochs)
- `--val_interval 1`: frequency (epoch) for calculating validation loss
- `--itr 5`: number of runs (for mean/std calculation)
- `--batch_size 32`: batch size
- `--learning_rate 1e-3` learning rate

## 4. Other Settings

All settings are available in `utils/configs.py` with detailed comments.

