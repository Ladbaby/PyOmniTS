# Code from: https://github.com/Ladbaby/PyOmniTS
import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from utils.ExpConfigs import ExpConfigs

'''
variable `configs` are ONLY determined by argparse, not .yaml files.

.yaml files are automatically saved and maintained as references:

- configs.yaml in training directory:

    Exact configs used in that training.

- *.yaml in "configs" folder

    Reference default configs for each model and dataset combination. Only automatically updated if new argument are added in argparse.

Note: Models and datasets only access the arguments they need.
'''

parser = argparse.ArgumentParser(description='PyOmniTS')

# basic config
parser.add_argument('--checkpoints', type=str, default='storage/results/', help='where to save model checkpoints in training. By default, results are organized under storage/results/DATASET_NAME/MODEL_NAME/MODEL_ID/SEQ_LEN_PRED_LEN/Y_md_HM/iter0')
parser.add_argument('--is_training', type=int, default=1, help='training or testing')
parser.add_argument('--model_id', type=str, default='GRU_D', help='By default, model_id is the same as model_name. However, if the model has multiple variants, then model_id will additionally contain more detailed information (backbone name, model size, etc). Unlike model_name, model_id only affects the storage location of experiment results (refer to --checkpoints) and yaml configs, so you can change it to whatever you like (e.g., GRU_D_d_model_512).')
parser.add_argument('--model_name', type=str, default='GRU_D', help='model name. PyOmniTS will try to construct the model from class Model in models/MODEL_NAME.py')
parser.add_argument('--task_name', type=str, choices=["long_term_forecast", "short_term_forecast", "imputation", "classification", "anomaly_detection", "representation_learning", "causal_discovery"], default='short_term_forecast', help='task name')

# dataset & data loader
parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment in regular time series forecasting datasets.")
parser.add_argument('--collate_fn', type=str, default=None, help='Name of the function as a custom collate_fn for dataloader. By default, datasets without collate_fn defined in data/data_provider/datasets/DATASET_NAME will use the default collate_fn of PyTorch. Refer to data/data_provider/data_factory.py for implementation detail.')
parser.add_argument('--dataset_file_name', type=str, default=None, help='Some datasets contains more than one files, then this argument may be ignored.')
parser.add_argument('--dataset_id', type=str, default='ETTm1', help='By default, dataset_id is the same as dataset_name. However, if the dataset has multiple subsets, then dataset_id will additionally contain dataset_subset_name. Unlike dataset_name, dataset_id only affects the storage location of experiment results (refer to --checkpoints), so you can change it to whatever you like (e.g., ETTm1_batch_size_32).')
parser.add_argument('--dataset_name', type=str, default='ETTm1', help='dataset name. PyOmniTS will try to construct the dataset from class Data in data/data_provider/datasets/DATASET_NAME.py')
parser.add_argument('--dataset_root_path', type=str, default='storage/datasets/ETT-small/', help='root path of the data file')
parser.add_argument('--dataset_subset_name', type=str, default=None, help='Some datasets may be composed of multiple subsets (e.g., UCR128).')
parser.add_argument('--embed', type=str, choices=["timeF", "fixed", "learned"], default='timeF', help='time features encoding. Only used in some regular time series datasets.')
parser.add_argument('--features', type=str, choices=['M', 'S', "MS"], default='M', help='M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate. WARNING: not thoroughly tested yet.')
parser.add_argument('--freq', type=str, choices=['s', 't', 'h', 'd', 'b', 'w', 'm', 'others'], default='h', help='freq for time features encoding in regular time series forecasting datasets, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--missing_rate', type=float, default=0., help="Manually mask out some observations. WARNING: not thoroughly tested yet.")
parser.add_argument('--target_variable_index', type=int, default=0, help='target variable index in datasets. Should not be used together with target_variable_name. WARNING: not thoroughly tested yet.')
parser.add_argument('--target_variable_name', type=str, default="OT", help='target variable name in regular time series forecasting datasets. Originally named as --target.')
parser.add_argument('--train_val_loader_drop_last', type=int, default=0, help="By default, train and val loader will not drop the last batch if the number of samples is not sufficient.")
parser.add_argument('--train_val_loader_shuffle', type=int, default=1, help="By default, train and val loader are shuffled.")

# forecasting task
parser.add_argument('--label_len', type=int, default=0, help='start token length. In other words, the length prepanded to the input of decoder. Only used in some regular time series forecasting models.')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length. Should be zero for tasks other than forecasting')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')

# classification task
parser.add_argument('--n_classes', type=int, default=2, help='number of classes')

# GPU
parser.add_argument('--gpu_id', type=int, default=0, help='primary gpu id, will be overwritten when use_multi_gpu is 1. Originally named as --gpu.')
parser.add_argument('--gpu_ids', type=str, default=None, help='string of device ids for multile gpus. Originally named as --devices. WARNING: not thoroughly tested yet. Currently, setting --use_multi_gpu 1 is enough.')
parser.add_argument('--use_gpu', type=int, default=1, help='Whether to use gpu.')
parser.add_argument('--use_multi_gpu', type=int, default=0, help='use multiple gpus, via huggingface accelerate library')

# training
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--itr', type=int, default=5, help='Number of runs for each experiment setting. By default, we run every experiment setting for 5 times using 5 different random seeds (2024~2028), then calculate the average and std of these 5 metrics.')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--loss', type=str, default='MSE', help='loss function. PyOmniTS will try to construct the loss function from class Loss in loss_fns/LOSS.py')
parser.add_argument('--lr_scheduler', type=str, choices=["ExponentialDecayLR", "ManualMilestonesLR", "DelayedStepDecayLR", "CosineAnnealingLR", "MultiStepLR", "StepLR"], default='DelayedStepDecayLR', help='learning rate scheduler. Originally named as --lradj')
parser.add_argument('--lr_scheduler_gamma', type=float, default=0.1, help='gamma for StepLR and MultiStepLR.')
parser.add_argument('--n_train_stages', type=int, default=1, help="Some models have multiple training stages, like pretraining + finetuning. e.g., --n_train_stages 2 will pass train_stage=1 and train_stage=2 to model during training.")
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--pretrained_checkpoint_file_name', type=str, default=None, help="file name of pretrained model's checkpoints, including file type extension")
parser.add_argument('--pretrained_checkpoint_root_path', type=str, default=None, help="Path to folder containing pretrained model's checkpoints")
parser.add_argument('--retain_graph', type=int, default=0, help='whether to retain compute graph in back propagation. Used in special models like HD_TTS.')
parser.add_argument('--sweep', type=int, default=0, help='whether to use weight & bias for hyperparameter searching')
parser.add_argument('--train_epochs', type=int, default=300, help='train epochs')
parser.add_argument('--val_interval', type=int, default=1, help='validation interval relative to training epochs')
parser.add_argument('--wandb', type=int, default=0, help='whether to use weight & bias for monitoring')

# testing
parser.add_argument('--checkpoints_test', type=str, default=None, help='folder where model checkpoint file is saved, for testing. If not given, Exp_Main.test() in exp/exp_main.py will try to find the latest checkpoint.')
parser.add_argument('--load_checkpoints_test', type=int, default=1, help='whether to load checkpoint during testing')
parser.add_argument('--save_arrays', type=int, default=0, help='whether to save model input and output as .npy files during testing, for later visualization')
parser.add_argument('--save_cache_arrays', type=int, default=0, help='whether to save model output (not input) as cache .npy files during every test iteration, such that the testing can recover from interruption. Designed for extremely slow models like diffusion models.')
parser.add_argument('--test_all', type=int, default=0, help='By default, we test the model on test set only. Setting this option to 1 will test the model on all samples from train, val, and test sets. Some datasets may not support this option yet.')
parser.add_argument('--test_dataset_statistics', type=int, default=0, help="Test dataset's statistics. See function Exp_Main.test() in exp/exp_main.py for implementation details.")
parser.add_argument('--test_flop', type=int, default=0, help='Test model flops. See utils/tools for implementation details.')
parser.add_argument('--test_gpu_memory', type=int, default=0, help="Test model's gpu memory usage. See utils/tools for implementation details.")
parser.add_argument('--test_train_time', type=int, default=0, help="Test model's training time. See utils/tools for implementation details.")

# model configs
# common
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--c_out', type=int, default=2, help='output size. Usually it is the same as --enc_in')
parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence for FreTS/TimeMixer model')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers. Used to replace n_layers in some model decoders.')
parser.add_argument('--d_model', type=int, default=512, help='size of latent embeddings')
parser.add_argument('--d_timesteps', type=int, default=1, help='UNUSED. Size of last dimension of `x_mark`/`y_mark`. Many Regular/Spatiotemporal datasets stack time in day, day in week, etc. along the last dimension. Others default to size 1.')
parser.add_argument('--dec_in', type=int, default=2, help='Number of variables in input time series (of decoder). Usually it is the same as --enc_in')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers. Used to replace n_layers in some model encoders.')
parser.add_argument('--embed_type', type=int, choices=[0, 1, 2, 3, 4], default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding; Only used in some regular time series forecasting models.')
parser.add_argument('--enc_in', type=int, default=2, help='Number of variables in input time series (of encoder). In most cases, it should be adjusted per dataset.') 
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--output_attention', type=int, default=0, help='DEPRECATED. output attention weight')
parser.add_argument('--hidden_layers', type=int, default=1, help='Number of hidden layers. WARNING: Will be replaced by --n_layers in the future, due to duplicated meaning.')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0. Only used in some regular time series forecasting models.')
parser.add_argument('--kernel_size', type=int, default=25, help='kernel size')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads (in attention)')
parser.add_argument('--n_layers', type=int, default=1, help='num of layers')
parser.add_argument('--node_dim', type=int, default=10, help='hidden dimension of nodes used in a few GNNs, like tPatchGNN')
parser.add_argument('--patch_len', type=int, default=12, help='patch length. Also used as period_len in some models (SparseTSF).')
parser.add_argument('--patch_stride', type=int, default=12, help='stride when splitting patches. Originally named as --stride.')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--revin_affine', type=int, default=0, help='RevIN-affine; True 1 False 0. Originally named as --affine.')
parser.add_argument('--scale_factor', type=int, default=2, help='scale factor for upsample. Only used in some regular time series forecasting models.')
parser.add_argument('--top_k', type=int, default=5, help='top k selection')
# CRU
parser.add_argument('--cru_bandwidth', type=int, default=3, help="Bandwidth for basis matrices A_k. b in paper")
parser.add_argument('--cru_num_basis', type=int, default=15, help="Number of basis matrices to use in transition model for locally-linear transitions. K in paper")
parser.add_argument('--cru_ts', type=float, default=1.0, help="Scaling factor of timestamps for numerical stability.")
# Informer
parser.add_argument('--informer_distil', type=int, default=1, help='whether to use distilling in encoder, using this argument means not using distilling')
# Latent ODE
parser.add_argument('--latent_ode_classif', type=int, default=0, help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")
parser.add_argument('--latent_ode_gen_layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")
parser.add_argument('--latent_ode_gru_units', type=int, default=100, help="Number of units per layer in each of GRU update networks")
parser.add_argument('--latent_ode_linear_classif', type=int, default=0, help="If using a classifier, use a linear classifier instead of 1-layer NN")
parser.add_argument('--latent_ode_rec_dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")
parser.add_argument('--latent_ode_rec_layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--latent_ode_units', type=int, default=100, help="Number of units per layer in ODE func")
parser.add_argument('--latent_ode_z0_encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")
# Mamba
parser.add_argument('--mamba_d_conv', type=int, default=4, help='conv kernel size for Mamba')
parser.add_argument('--mamba_expand', type=int, default=2, help='expansion factor for Mamba')
# mTAN
parser.add_argument('--mtan_alpha', type=float, default=100., help='In classification task, loss is calculated as recon_loss + self.alpha * ce_loss')
parser.add_argument('--mtan_num_ref_points', type=int, default=8, help='number of reference points, originally chosen in [8, 16, 32, 64, 128]')
# NeuralFlows
parser.add_argument('--neuralflows_flow_layers', type=int, default=1, help='Number of flow layers')
parser.add_argument('--neuralflows_flow_model', type=str, default='coupling', help='Type of NeuralFlows model', choices=['coupling', 'resnet', 'gru'])
parser.add_argument('--neuralflows_latents', type=int, default=20, help='Size of the latent state')
parser.add_argument('--neuralflows_time_hidden_dim', type=int, default=1, help='Number of time features (only for Fourier)')
parser.add_argument('--neuralflows_time_net', type=str, default='TimeLinear', help='Name of time net', choices=['TimeFourier', 'TimeFourierBounded', 'TimeLinear', 'TimeTanh'])
# Nonstationary Transformer
parser.add_argument('--nonstationarytransformer_p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
parser.add_argument('--nonstationarytransformer_p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
# PatchTST
parser.add_argument('--patchtst_decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--patchtst_fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--patchtst_head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patchtst_padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--patchtst_subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
# PrimeNet
parser.add_argument('--primenet_pooling', type=str, default='ave', help='[ave, att, bert]: What pooling to use to aggregate the model output sequence representation for different tasks.')
# TimeMixer
parser.add_argument('--timemixer_decomp_method', type=str, default='moving_avg', help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--timemixer_down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--timemixer_down_sampling_method', type=str, default="avg", help='down sampling method, only support avg, max, conv')
parser.add_argument('--timemixer_use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
# tPatchGNN
parser.add_argument('--tpatchgnn_te_dim', type=int, default=10, help="Number of units for time encoding")

# Used to be compatible with ipython. Never used
parser.add_argument("--f", help="a dummy argument to fool ipython", default="1")

configs = ExpConfigs(**vars(parser.parse_args())) # enable type hints

# .yaml reference file maintainance
yaml_configs_path_deprecated = Path(f"configs/{configs.model_name}/{configs.dataset_name}.yaml") # backward compatibility
yaml_configs_path = Path(f"configs/{configs.model_name}/{configs.model_id}/{configs.dataset_name}.yaml")
if yaml_configs_path.exists():
    with open(yaml_configs_path, 'r', encoding="utf-8") as stream:
        try:
            yaml_configs: dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f"utils/configs.py: Exception when parsing {yaml_configs_path}: {exc}")
            exit(1)
    if yaml_configs is not None:
        # update yaml only if new args are added in argparse
        if_update = False
        for key, value in configs.__dict__.items():
            if key not in yaml_configs.keys():
                if_update = True
                yaml_configs[key] = value

        if if_update:
            with open(yaml_configs_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_configs, f, default_flow_style=False)
else:
    Path(f"configs/{configs.model_name}/{configs.model_id}").mkdir(parents=True, exist_ok=True)
    if yaml_configs_path_deprecated.exists():
        # migrate from deprecated folder structure to new one
        yaml_configs_path_deprecated.replace(yaml_configs_path) # will overwrite if destination exists
    else:
        # save yaml if not exist
        with open(yaml_configs_path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(configs), f, default_flow_style=False)
