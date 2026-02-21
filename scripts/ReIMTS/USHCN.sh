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

seq_len=150
for pred_len in 3; do
    for ts_backbone_name in \
        GraFITi \
        PrimeNet \
        mTAN \
        TimeCHEAT \
        GRU_D \
        Raindrop
    do
        model_id="$model_name"_$ts_backbone_name
        case $ts_backbone_name in
            GraFITi) 
                reimts_pad_time_emb=1
                ts_backbone_overwrite_config_list="d_model n_layers n_heads"
                patch_len_list=100
                case $pred_len in
                    50)
                        d_model=128
                        n_layers=2
                        n_heads=1
                    ;;
                    *)
                        d_model=32
                        n_layers=1
                        n_heads=4
                esac
                learning_rate=1e-3
                ;;
            PrimeNet)
                reimts_pad_time_emb=1
                ts_backbone_overwrite_config_list="d_model n_layers n_heads"
                patch_len_list=100
                d_model=128
                n_layers=2
                n_heads=1
                learning_rate=1e-4
                ;;
            mTAN)
                reimts_pad_time_emb=0
                ts_backbone_overwrite_config_list="d_model n_layers n_heads"
                patch_len_list=100
                d_model=128
                n_layers=2
                n_heads=1
                learning_rate=1e-3
                ;;
            TimeCHEAT) 
                reimts_pad_time_emb=1
                ts_backbone_overwrite_config_list="d_model n_layers n_heads"
                patch_len_list=100
                d_model=64
                n_layers=4
                n_heads=4
                learning_rate=1e-3
                ;;
            GRU_D) 
                reimts_pad_time_emb=1
                ts_backbone_overwrite_config_list="d_model n_layers n_heads"
                patch_len_list=100
                d_model=32
                n_layers=2
                n_heads=4
                learning_rate=1e-3
                ;;
            Raindrop) 
                reimts_pad_time_emb=1
                ts_backbone_overwrite_config_list="d_model n_layers n_heads"
                patch_len_list=100
                d_model=16
                n_layers=2
                n_heads=4
                learning_rate=1e-3
                ;;
            HyperIMTS) 
                reimts_pad_time_emb=1
                ts_backbone_overwrite_config_list="d_model n_layers n_heads"
                patch_len_list=100
                d_model=64
                n_layers=4
                n_heads=1
                learning_rate=1e-3
                ;;
        esac
        $launch_command main.py \
            --is_training 1 \
            --reimts_pad_time_emb $reimts_pad_time_emb \
            --ts_backbone_name $ts_backbone_name \
            --ts_backbone_overwrite_config_list $ts_backbone_overwrite_config_list \
            --patch_len_list $patch_len_list \
            --d_model $d_model \
            --n_layers $n_layers \
            --n_heads $n_heads \
            --loss "MSE" \
            --collate_fn "collate_fn_fractal" \
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
            --batch_size 16 \
            --learning_rate $learning_rate
    done
done

