use_multi_gpu=0
if [ $use_multi_gpu -eq 0 ]; then
    launch_command="python"
else
    launch_command="accelerate launch"
fi

model_name="$(basename "$(dirname "$(readlink -f "$0")")")" # folder name

dataset_root_path=storage/datasets/OpenMIC
dataset_file_name=openmic-2018.npz
model_id_name=$model_name
dataset_name=$(basename "$0" .sh) # file name

seq_len=$((10))
for pred_len in 0; do
    for batch_size in 128; do
        for i in $(seq 1 1); do
        $launch_command main.py \
            --is_training 1 \
            --factor 3 \
            --d_model 32 \
            --d_ff 32 \
            --n_classes 20 \
            --task_name "classification" \
            --loss "CrossEntropyLoss" \
            --use_multi_gpu $use_multi_gpu \
            --dataset_root_path $dataset_root_path \
            --dataset_file_name $dataset_file_name \
            --model_id $model_id_name'_'$seq_len'_'$pred_len \
            --model_name $model_name \
            --dataset_name $dataset_name \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 128 \
            --c_out 128 \
            --dec_in 128 \
            --train_epochs 300 \
            --patience 10 \
            --val_interval 1 \
            --itr 1 \
            --batch_size $batch_size \
            --learning_rate 1e-3
        done
    done
done

