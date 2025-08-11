use_multi_gpu=0
if [ $use_multi_gpu -eq 0 ]; then
    launch_command="python"
else
    launch_command="accelerate launch"
fi

model_name="$(basename "$(dirname "$(readlink -f "$0")")")" # folder name

dataset_root_path=storage/datasets/electricity
model_id=$model_name
dataset_name=$(basename "$0" .sh) # file name

seq_len=96
label_len=48
for pred_len in 24 48 168 336 720; do
    $launch_command main.py \
    --is_training 1 \
    --loss "MSE" \
    --d_model 128 \
    --n_layers 2 \
    --n_heads 4 \
    --task_name "long_term_forecast" \
    --use_multi_gpu $use_multi_gpu \
    --dataset_root_path $dataset_root_path \
    --dataset_file_name "electricity.csv" \
    --model_id $model_id \
    --model_name $model_name \
    --dataset_name $dataset_name \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --train_epochs 300 \
    --patience 10 \
    --val_interval 1 \
    --itr 5 \
    --batch_size 1 \
    --learning_rate 0.001
done

