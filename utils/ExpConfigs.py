from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class ExpConfigs:
    '''
    dataclass for argparse typo check, making life easier

    Make sure to update this dataclass after adding new args in argparse
    '''
    @classmethod
    def get_sweep_values(cls, attr_name: str) -> Optional[list]:
        for field_info in cls.__dataclass_fields__.values():
            if field_info.name == attr_name:
                return field_info.metadata.get('sweep')
        return None

    # basic config
    task_name: str
    is_training: int
    model_id: str
    model_name: str
    checkpoints: str

    # dataset & data loader
    dataset_name: str
    dataset_root_path: str
    dataset_file_name: str | None
    features: str
    target_variable_name: str
    target_variable_index: int
    freq: str
    collate_fn: str
    augmentation_ratio: int
    missing_rate: float
    train_val_loader_shuffle: int
    train_val_loader_drop_last: int

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
    gpu_ids: str | None

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
    checkpoints_test: str | None
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
    d_model: int = field(metadata={"sweep": [32, 64, 128, 256]})
    d_timesteps: int
    n_heads: int
    n_layers: int = field(metadata={"sweep": [1, 2, 3, 4]})
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
    seq_len_max_irr: int | None = None # maximum number of observations along time dimension of x, set in irregular time series datasets
    pred_len_max_irr: int | None = None # maximum number of observations along time dimension of y, set in irregular time series datasets
    patch_len_max_irr: int | None = None # maximum number of observations along time dimension in a patch of x, set in irregular time series datasets
    subfolder_train: str = "" # timestamp of training in format %Y_%m%d_%H%M
    itr_i: int = 0 # current training iteration. [0, itr-1]

class ExpConfigsTracker:
    """Wrapper that tracks which ExpConfigs attributes are accessed"""
    
    def __init__(self, configs: ExpConfigs):
        object.__setattr__(self, '_config', configs)
        object.__setattr__(self, '_accessed_attrs', set())
    
    def __getattr__(self, name: str) -> Any:
        if hasattr(self._config, name):
            self._accessed_attrs.add(name)
            return getattr(self._config, name)
        raise AttributeError(f"'{type(self._config).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._accessed_attrs.add(name)
            setattr(self._config, name, value)
    
    def get_accessed_attributes(self) -> set[str]:
        """Return set of accessed attribute names"""
        return self._accessed_attrs.copy()
    
    def get_unused_attributes(self) -> set[str]:
        """Return set of unused attribute names"""
        all_attrs = {field.name for field in self._config.__dataclass_fields__.values()}
        return all_attrs - self._accessed_attrs
    
    def print_access_report(self):
        """Print a report of accessed vs unused attributes"""
        accessed = self.get_accessed_attributes()
        unused = self.get_unused_attributes()
        
        print("=== ExpConfigs Access Report ===")
        print(f"Accessed attributes ({len(accessed)}):")
        for attr in sorted(accessed):
            print(f"  ✓ {attr}")