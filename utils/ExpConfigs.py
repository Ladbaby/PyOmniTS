from dataclasses import dataclass

@dataclass
class ExpConfigs:
    '''
    dataclass for argparse typo check, making life easier

    Make sure to update this dataclass after adding new args in argparse
    '''
    # basic config
    task_name: str
    is_training: int
    model_id: str
    model_name: str
    checkpoints: str

    # dataset & data loader
    dataset_name: str
    dataset_root_path: str
    dataset_file_name: str
    features: str
    target_variable_name: str
    target_variable_index: int
    freq: str
    collate_fn: str
    augmentation_ratio: int
    missing_rate: float

    # forecasting task
    seq_len: int
    label_len: int
    pred_len: int

    # classification task
    n_classes: int

    # GPU
    use_gpu: int
    gpu_id: int
    use_multi_gpu: int
    gpu_ids: str

    # training
    wandb: int
    sweep: int
    val_interval: int
    num_workers: int
    itr: int
    train_epochs: int
    batch_size: int
    patience: int
    learning_rate: float
    loss: str
    lr_scheduler: str
    pretrained_checkpoint_root_path: str
    pretrained_checkpoint_file_name: str
    n_train_stages: str
    retain_graph: int

    # testing
    checkpoints_test: str
    test_all: int
    test_flop: int
    test_train_time: int
    test_gpu_memory: int
    test_zero_shot: int
    test_dataset_statistics: int
    save_arrays: int
    load_checkpoints_test: int

    # model configs
    # common
    patch_len: int
    patch_stride: int
    revin: int
    revin_affine: int
    kernel_size: int
    individual: int
    channel_independence: int
    scale_factor: int
    top_k: int
    embed_type: int
    enc_in: int
    dec_in: int
    c_out: int
    d_model: int
    d_timesteps: int
    n_heads: int
    n_layers: int
    e_layers: int
    d_layers: int
    hidden_layers: int
    d_ff: int
    moving_avg: int
    factor: int
    dropout: float
    embed: str
    activation: str
    output_attention: int
    node_dim: int
    # PatchTST
    patchtst_fc_dropout: float
    patchtst_head_dropout: float
    patchtst_padding_patch: str
    patchtst_subtract_last: int
    patchtst_decomposition: int
    # Mamba
    mamba_d_conv: int
    mamba_expand: int
    # Latent ODE
    latent_ode_units: int
    latent_ode_gen_layers: int
    latent_ode_rec_layers: int
    latent_ode_z0_encoder: str
    latent_ode_rec_dims: int
    latent_ode_gru_units: int
    latent_ode_classif: int
    latent_ode_linear_classif: int
    # CRU
    cru_num_basis: int
    cru_bandwidth: int
    cru_ts: float
    # NeuralFlows
    neuralflows_flow_model: str
    neuralflows_flow_layers: int
    neuralflows_latents: int
    neuralflows_time_net: str
    neuralflows_time_hidden_dim: int
    # PrimeNet
    primenet_pooling: str
    # mTAN
    mtan_num_ref_points: int
    mtan_alpha: float
    # TimeMixer
    timemixer_decomp_method: str
    timemixer_use_norm: int
    timemixer_down_sampling_layers: int
    timemixer_down_sampling_method: str
    # Nonstationary Transformer
    nonstationarytransformer_p_hidden_dims: list
    nonstationarytransformer_p_hidden_layers: int
    # Informer
    informer_distil: int
    # tPatchGNN
    tpatchgnn_te_dim: int

    # Used to be compatible with ipython. Never used
    f: int = 1

    # args not presented in argparse
    seq_len_max_irr: int = None # maximum number of observations along time dimension of x, set in irregular time series datasets
    pred_len_max_irr: int = None # maximum number of observations along time dimension of y, set in irregular time series datasets
    patch_len_max_irr: int = None # maximum number of observations along time dimension in a patch of x, set in irregular time series datasets
