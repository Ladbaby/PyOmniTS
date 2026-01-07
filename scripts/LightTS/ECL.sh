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

seq_len=96
label_len=48
for pred_len in 96 192 336 720; do
    $launch_command main.py \
    --is_training 1 \
    --patch_len 24 \
    --loss "MSE" \
    --task_name "long_term_forecast" \
    --use_multi_gpu $use_multi_gpu \
    --dataset_root_path $dataset_root_path \
    --dataset_file_name "electricity.csv" \
    --model_id $model_id \
    --model_name $model_name \
    --dataset_name $dataset_name \
    --dataset_id $dataset_id \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --enc_in $n_variables \
    --dec_in $n_variables \
    --c_out $n_variables \
    --train_epochs 300 \
    --patience 10 \
    --itr 5 \
    --batch_size 32 \
    --learning_rate 1e-4
done

