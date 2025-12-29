# Code from: https://github.com/Ladbaby/PyOmniTS
import datetime
import importlib
import pprint
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from exp.exp_main import Exp_Main
from utils.configs import get_configs
from utils.ExpConfigs import ExpConfigs
from utils.globals import accelerator, logger


class ExperimentRunner:
    """
    Invoke Exp_Main with automatic batch size reduction.
    Core functions:
    1. run_training(): train + val + test (when --is_training 1)
    2. run_test_only(): test (when --is_training 0)
    3. run_sweep(): To be invoked by SweepManager.run_sweep() (when --sweep 1)
    """
    
    def __init__(self, configs: ExpConfigs, hyperparameters_sweep: dict = None):
        self.configs = configs
        self.hyperparameters_sweep = hyperparameters_sweep
        self.exp = None
    
    def _setup_random_seeds(self, iteration: int):
        """Set random seeds for reproducibility."""
        seed = 2024 + iteration
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def _create_output_path(self) -> Path:
        """Create and return the output path for checkpoints."""
        path = (Path(self.configs.checkpoints) / 
                self.configs.dataset_name / 
                self.configs.dataset_id / 
                self.configs.model_name / 
                self.configs.model_id / 
                f"{self.configs.seq_len}_{self.configs.pred_len}" / 
                self.configs.subfolder_train / 
                f"iter{self.configs.itr_i}")
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def _save_configs(self, path: Path):
        """
        Save training configuration to YAML file under the training directory.
        Note: YAML will also be updated under 'configs/' folder, which is done by get_configs() in utils/configs.py
        """
        with open(path / "configs.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self.configs), f, default_flow_style=False)
    
    def _init_wandb(self, path: Path):
        """
        Major logics:
        1. Initialize Weights & Biases tracking if enabled.
        2. Overwrite self.configs with hyperparameter settings when --sweep 1.
        """
        if (self.configs.wandb and accelerator.is_main_process) or self.configs.sweep:
            assert self.hyperparameters_sweep is not None, \
                "Please provide 'hyperparameters_sweep' when using --sweep 1."
            
            import wandb
            wandb.init(
                project="YOUR_PROJECT_NAME",
                config={
                    "model_name": self.configs.model_name,
                    "model_id": self.configs.model_id,
                    "dataset_name": self.configs.dataset_name,
                    "seq_len": self.configs.seq_len,
                    "pred_len": self.configs.pred_len,
                    "learning_rate": self.configs.learning_rate,
                    "batch_size": self.configs.batch_size
                },
                dir=path
            )
            
            # Overwrite hyperparameters when sweeping
            if self.configs.sweep:
                for attr_name in self.hyperparameters_sweep.keys():
                    setattr(self.configs, attr_name, getattr(wandb.config, attr_name))
    
    def _run_with_batch_reduction(self, func, operation_name: str):
        """
        Execute a function with automatic batch size reduction on CUDA OOM.
        
        Args:
            func: Function to execute (should be train or test method)
            operation_name: Name of the operation (for logging)
        """
        original_batch_size = self.configs.batch_size
        
        while self.configs.batch_size >= 1:
            try:
                return func()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                # Check if it's a CUDA OOM-related error
                is_oom = isinstance(e, torch.cuda.OutOfMemoryError)
                is_cublas_oom = isinstance(e, RuntimeError) and (
                    "CUBLAS_STATUS_ALLOC_FAILED" in str(e) or
                    "CUDA out of memory" in str(e)
                )
                
                if not (is_oom or is_cublas_oom):
                    # Not an OOM error, re-raise it
                    raise
                elif is_cublas_oom:
                    logger.warning(f"PyOmniTS has regarded the following error as CUDA OOM error: {e}")
                
                torch.cuda.empty_cache()
                
                if self.configs.batch_size == 1:
                    logger.exception(
                        f"CUDA OOM error during {operation_name} even with batch_size=1. "
                        f"Operation aborted."
                    )
                    exit(1)
                
                new_batch_size = max(1, self.configs.batch_size // 2)
                logger.error(
                    f"CUDA OOM error during {operation_name}! "
                    f"Reducing batch_size from {self.configs.batch_size} to {new_batch_size}"
                )
                self.configs.batch_size = new_batch_size
        
        # Restore original batch size for subsequent operations
        self.configs.batch_size = original_batch_size
    
    def train(self) -> Exp_Main:
        """
        Train the model with automatic batch size reduction.
        Major logics:
        1. Invoke Exp_Main.train() once, which also calls Exp_Main.vali()
        """
        path = self._create_output_path()
        logger.info(f"Training iter{self.configs.itr_i} save to: {path}")
        
        self._save_configs(path)
        self._init_wandb(path)
        accelerator.project_configuration.set_directories(project_dir=path)
        
        self.exp = Exp_Main(self.configs)
        self._run_with_batch_reduction(self.exp.train, "training")
        return self.exp
    
    def test(self):
        """
        Test the model with automatic batch size reduction.
        Major logics:
        1. Invoke Exp_Main.test() once.
        """
        if self.exp is None:
            self.exp = Exp_Main(self.configs)
        
        self._run_with_batch_reduction(self.exp.test, "testing")
        torch.cuda.empty_cache()
    
    def run_sweep(self):
        """
        Run hyperparameter sweep.
        1. Force overwrite some configs.
        2. Invoke self.train() & self.test() once.
        """
        logger.info('>>>>>>> sweeping start <<<<<<<')
        
        self.configs.subfolder_train = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
        self.configs.wandb = 1
        self.configs.itr = 1
        self.configs.itr_i = 0
        
        logger.debug('wandb=1: Weight & Bias logging is automatically enabled')
        logger.debug('itr=1: training iteration is automatically overwritten to 1')
        
        self._setup_random_seeds(0)
        self.train()
        self.test()
    
    def run_training(self):
        """
        Major logics:
        1. Run self.train() for 'configs.itr' times using different random seeds.
        2. Invoke self.test() once.
        """
        self.configs.subfolder_train = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
        
        for i in range(self.configs.itr):
            self.configs.itr_i = i
            self._setup_random_seeds(i)
            self.train()
            torch.cuda.empty_cache()
        
        self.test()
    
    def run_test_only(self):
        """Run testing only."""
        self.test()


class SweepManager:
    """
    Major logics:
    1. Discover hyperparameters to be searched.
    2. SweepManager.run_sweep() -> ExperimentRunner.run_sweep()
    """
    
    def __init__(self, configs: ExpConfigs):
        self.configs = configs
        self.hyperparameters_sweep = {}
    
    def discover_hyperparameters(self) -> dict:
        """
        Discover which hyperparameters the model accesses, and get their search spaces from utils/ExpConfigs.py
        """
        from utils.ExpConfigs import ExpConfigsTracker
        
        configs_tracker = ExpConfigsTracker(self.configs)
        model_module = importlib.import_module("models." + self.configs.model_name)
        model = model_module.Model(configs_tracker)
        del model
        
        accessed_configs = configs_tracker.get_accessed_attributes()
        max_count = 1
        
        for accessed_config in accessed_configs:
            try:
                ref_values = self.configs.get_sweep_values(accessed_config)
                if ref_values and isinstance(ref_values, list):
                    self.hyperparameters_sweep[accessed_config] = {"values": ref_values}
                    max_count *= len(ref_values)
            except Exception:
                continue
        
        if not self.hyperparameters_sweep:
            logger.error("No hyperparameter to be searched, stopping now..")
            logger.debug(f"{self.configs.model_name} access these attributes in ExpConfigs:")
            configs_tracker.print_access_report()
            logger.debug(
                "Possible reasons: (1) The model does not access any hyperparameters; "
                "(2) The accessed hyperparameters have not set their metadata properly. "
                "Check the ExpConfigs class. Example: "
                "d_model: int = field(metadata={'sweep': [32, 64, 128, 256]})"
            )
            exit(0)
        
        return self.hyperparameters_sweep, max_count
    
    def create_sweep(self) -> str:
        """Create and return sweep ID."""
        hyperparameters, max_count = self.discover_hyperparameters()
        
        sweep_method = "grid" if max_count <= 16 else "bayes"
        max_count = min(max_count, 16)
        
        logger.info(
            f"{len(hyperparameters)} hyperparameters and {max_count} runs "
            f'using "{sweep_method}" as the sweep method:\n'
            f"{pprint.pformat(hyperparameters)}"
        )
        
        import wandb
        sweep_configuration = {
            "method": sweep_method,
            "metric": {"goal": "minimize", "name": "loss_val_best"},
            "parameters": hyperparameters
        }
        
        temp_file_path = "storage/tmp.txt"
        
        if accelerator.is_main_process:
            sweep_id = wandb.sweep(
                sweep=sweep_configuration, 
                project="YOUR_PROJECT_NAME"
            )
            with open(temp_file_path, mode='w', encoding="utf-8") as f:
                f.write(sweep_id)
        
        accelerator.wait_for_everyone()
        
        with open(temp_file_path, mode='r', encoding="utf-8") as f:
            sweep_id = f.readline()
        
        return sweep_id, max_count
    
    def run_sweep(self):
        """
        Execute the hyperparameter sweep.
        Major logics:
        1. self.create_sweep() -> self.discover_hyperparameters()
        2. wandb.agent() -> ExperimentRunner.run_sweep()
        """
        sweep_id, max_count = self.create_sweep()
        
        def sweep_main():
            runner = ExperimentRunner(self.configs, self.hyperparameters_sweep)
            runner.run_sweep()
        
        import wandb
        wandb.agent(
            sweep_id,
            function=sweep_main,
            project="YOUR_PROJECT_NAME",
            count=max_count
        )


def main(configs: ExpConfigs, hyperparameters_sweep: dict = None):
    """
    Major logics:
    ├── --is_training 1 -> ExperimentRunner.run_training(): train+val+test
    └── otherwise -> ExperimentRunner.run_test_only(): test
    """
    runner = ExperimentRunner(configs, hyperparameters_sweep)
    
    if configs.is_training:
        runner.run_training()
    else:
        runner.run_test_only()


if __name__ == "__main__":
    '''
    Major logics:
    ├── --sweep 1 -> SweepManager().run_sweep(): search hyperparameters
    └── otherwise -> main(): train/val/test
    '''
    configs: ExpConfigs = get_configs()
    
    try:
        if not configs.sweep:
            main(configs=configs)
        else:
            sweep_manager = SweepManager(configs)
            sweep_manager.run_sweep()
    
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print("\nProcess interrupted...")
    
    except Exception as e:
        if accelerator.is_main_process:
            logger.exception(f"{e}", stack_info=True)
            exit(1)