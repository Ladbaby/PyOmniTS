# Code from: https://github.com/Ladbaby/PyOmniTS
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
    checkpoints: str
    is_training: int
    model_id: str
    model_name: str
    task_name: str

    # dataset & data loader
    augmentation_ratio: int
    collate_fn: str | None
    dataset_file_name: str | None
    dataset_id: str
    dataset_name: str
    dataset_root_path: str
    dataset_subset_name: str | None
    embed: str
    features: str
    freq: str
    missing_rate: float
    target_variable_index: int
    target_variable_name: str
    train_val_loader_drop_last: int
    train_val_loader_shuffle: int

    # forecasting task
    label_len: int
    pred_len: int
    seq_len: int

    # classification task
    n_classes: int

    # GPU
    allow_tf32: int
    gpu_id: int
    gpu_ids: str | None
    use_gpu: int
    use_multi_gpu: int

    # training
    batch_size: int
    itr: int
    learning_rate: float
    loss: str
    lr_scheduler: str
    lr_scheduler_gamma: float
    n_train_stages: str
    num_workers: int
    patience: int
    pretrained_checkpoint_file_name: str | None
    pretrained_checkpoint_root_path: str | None
    retain_graph: int
    sweep: int
    train_epochs: int
    val_interval: int
    wandb: int

    # testing
    checkpoints_test: str | None
    load_checkpoints_test: int
    save_arrays: int
    save_cache_arrays: int
    test_all: int
    test_dataset_statistics: int
    test_flop: int
    test_gpu_memory: int
    test_train_time: int

    # model configs
    # common
    activation: str
    c_out: int
    channel_independence: int
    d_ff: int = field(metadata={"sweep": [16, 32, 64, 128, 256, 512, 2048]})
    d_layers: int = field(metadata={"sweep": [1, 2]})
    d_model: int = field(metadata={"sweep": [16, 32, 64, 128, 256, 512]})
    d_timesteps: int
    dec_in: int
    dropout: float = field(metadata={"sweep": [0.0, 0.1, 0.3, 0.4, 0.5]})
    e_layers: int = field(metadata={"sweep": [1, 2, 3, 4, 8]})
    embed_type: int
    enc_in: int
    factor: int
    output_attention: int
    hidden_layers: int
    individual: int
    kernel_size: int = field(metadata={"sweep": [2, 3, 4, 5]})
    moving_avg: int
    n_heads: int = field(metadata={"sweep": [1, 4, 8]})
    n_layers: int = field(metadata={"sweep": [1, 2, 3, 4]})
    n_patches_list: list[int]
    node_dim: int
    patch_len: int
    patch_len_list: list[int]
    patch_stride: int
    revin: int
    revin_affine: int
    scale_factor: int
    top_k: int = field(metadata={"sweep": [3, 5]})
    # Adaptor
    ts_backbone_name: str
    ts_backbone_overwrite_config_list: list[str]
    # CRU
    cru_bandwidth: int
    cru_num_basis: int
    cru_ts: float = field(metadata={"sweep": [0.2, 0.3]})
    # Informer
    informer_distil: int
    # Latent ODE
    latent_ode_classif: int
    latent_ode_gen_layers: int = field(metadata={"sweep": [2, 3]})
    latent_ode_gru_units: int = field(metadata={"sweep": [50, 100]})
    latent_ode_linear_classif: int
    latent_ode_rec_dims: int = field(metadata={"sweep": [30, 40, 100]})
    latent_ode_rec_layers: int = field(metadata={"sweep": [2, 3, 4]})
    latent_ode_units: int = field(metadata={"sweep": [50, 300, 500]})
    latent_ode_z0_encoder: str
    # Mamba
    mamba_d_conv: int
    mamba_expand: int
    # mTAN
    mtan_alpha: float = field(metadata={"sweep": [5., 100.]})
    mtan_num_ref_points: int = field(metadata={"sweep": [8, 16, 32, 64, 128]})
    # NeuralFlows
    neuralflows_flow_layers: int = field(metadata={"sweep": [1, 2, 4, 16]})
    neuralflows_flow_model: str
    neuralflows_latents: int = field(metadata={"sweep": [15, 20]})
    neuralflows_time_hidden_dim: int
    neuralflows_time_net: str
    # Nonstationary Transformer
    nonstationarytransformer_p_hidden_dims: list = field(metadata={"sweep": [[8, 8], [16, 16], [32, 32], [64, 64], [128, 128], [256, 256]]})
    nonstationarytransformer_p_hidden_layers: int = field(metadata={"sweep": [2, 4]})
    # PatchTST
    patchtst_decomposition: int
    patchtst_fc_dropout: float
    patchtst_head_dropout: float
    patchtst_padding_patch: str
    patchtst_subtract_last: int
    # PrimeNet
    primenet_pooling: str
    # ReIMTS
    reimts_pad_time_emb: int
    # TimeMixer
    timemixer_decomp_method: str
    timemixer_down_sampling_layers: int = field(metadata={"sweep": [1, 3]})
    timemixer_down_sampling_method: str
    timemixer_use_norm: int
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

    @property
    def __dict__(self):
        return self._config.__dict__
