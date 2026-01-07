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

seq_len=36
label_len=18
for pred_len in 12 24 36 48; do
    $launch_command main.py \
    --is_training 1 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --loss "MSE" \
    --task_name "long_term_forecast" \
    --use_multi_gpu $use_multi_gpu \
    --dataset_root_path $dataset_root_path \
    --dataset_file_name "national_illness.csv" \
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
    --learning_rate 1e-3
done

