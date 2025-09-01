use_multi_gpu=0
if [ $use_multi_gpu -eq 0 ]; then
    launch_command="python"
else
    launch_command="accelerate launch"
fi

model_name="$(basename "$(dirname "$(readlink -f "$0")")")" # folder name

dataset_root_path=storage/datasets/MIMIC_IV
model_id=$model_name
dataset_name=$(basename "$0" .sh) # file name

seq_len=2160
for pred_len in 3; do
    $launch_command main.py \
    --is_training 1 \
    --d_model 128 \
    --collate_fn "collate_fn_patch" \
    --patch_len 360 \
    --n_heads 1 \
    --pretrained_checkpoint_root_path "storage/pretrained/PrimeNet" \
    --pretrained_checkpoint_file_name "87623.h5" \
    --loss "ModelProvidedLoss" \
    --use_multi_gpu $use_multi_gpu \
    --dataset_root_path $dataset_root_path \
    --model_id $model_id \
    --model_name $model_name \
    --dataset_name $dataset_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 100 \
    --dec_in 100 \
    --c_out 100 \
    --train_epochs 300 \
    --patience 10 \
    --val_interval 1 \
    --itr 5 \
    --batch_size 32 \
    --learning_rate 0.0001
done

