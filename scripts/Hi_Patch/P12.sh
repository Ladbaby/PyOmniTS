use_multi_gpu=1
if [ $use_multi_gpu -eq 0 ]; then
    launch_command="python"
else
    launch_command="accelerate launch"
fi

model_name="$(basename "$(dirname "$(readlink -f "$0")")")" # folder name

dataset_root_path=storage/datasets/P12
model_id=$model_name
dataset_name=$(basename "$0" .sh) # file name

seq_len=$((36))
for pred_len in 3; do
    for batch_size in 32; do
        for i in $(seq 1 1); do
        $launch_command main.py \
            --is_training 1 \
            --loss "MSE" \
            --collate_fn "collate_fn_patch" \
            --n_heads 1 \
            --n_layers 1 \
            --d_model 64 \
            --patch_len 6 \
            --patch_stride 6 \
            --use_multi_gpu $use_multi_gpu \
            --dataset_root_path $dataset_root_path \
            --model_id $model_id \
            --model_name $model_name \
            --dataset_name $dataset_name \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 36 \
            --c_out 36 \
            --train_epochs 300 \
            --patience 5 \
            --val_interval 1 \
            --itr 5 \
            --batch_size $batch_size \
            --learning_rate 0.001
        done
    done
done

