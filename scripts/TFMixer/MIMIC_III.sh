use_multi_gpu=0
if [ $use_multi_gpu -eq 0 ]; then
    launch_command="python"
else
    launch_command="accelerate launch"
fi

. "$(dirname "$(readlink -f "$0")")/../globals.sh"

dataset_name=$(basename "$0" .sh)
dataset_subset_name=""
dataset_id=$dataset_name
get_dataset_info "$dataset_name" "$dataset_subset_name"

model_name="$(basename "$(dirname "$(readlink -f "$0")")")"
model_id=$model_name

seq_len=72
for pred_len in 3; do
    $launch_command main.py \
    --is_training 1 \
    --d_model 64 \
    --dropout 0 \
    --n_patches_list 8 \
    --tpatchgnn_te_dim 16 \
    --n_layers 1 \
    --cru_num_basis 32 \
    --factor 4 \
    --top_k 6 \
    --revin 0 \
    --mtan_alpha 0.1 \
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
    --c_out $n_variables \
    --dec_in $n_variables \
    --train_epochs 300 \
    --patience 10 \
    --val_interval 1 \
    --itr 5 \
    --batch_size 32 \
    --learning_rate 1e-3
done
