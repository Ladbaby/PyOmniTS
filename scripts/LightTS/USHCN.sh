use_multi_gpu=0
if [ $use_multi_gpu -eq 0 ]; then
    launch_command="python"
else
    launch_command="accelerate launch"
fi

model_name="$(basename "$(dirname "$(readlink -f "$0")")")" # folder name

dataset_root_path=storage/datasets/USHCN
model_id=$model_name
dataset_name=$(basename "$0" .sh) # file name

seq_len=$((150))
for pred_len in 3; do
    for batch_size in 16; do
        $launch_command main.py \
            --is_training 1 \
            --patch_len 50 \
            --loss "MSE" \
            --use_multi_gpu $use_multi_gpu \
            --dataset_root_path $dataset_root_path \
            --model_id $model_id \
            --model_name $model_name \
            --dataset_name $dataset_name \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 5 \
            --dec_in 5 \
            --c_out 5 \
            --train_epochs 300 \
            --patience 10 \
            --val_interval 1 \
            --itr 5 \
            --batch_size $batch_size \
            --learning_rate 0.0001
    done
done

