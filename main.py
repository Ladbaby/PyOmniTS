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
from utils.configs import configs
from utils.globals import accelerator, logger

hyperparameters_sweep: dict[str, dict[str, list]] = {}

def main():
    # random seed
    fix_seed_list = range(2024, 2024 + configs.itr)

    configs.use_gpu = True if torch.cuda.is_available() and configs.use_gpu else False

    Exp = Exp_Main

    def start_exp_train() -> Exp_Main:
        # save training config file for reference
        path = Path(configs.checkpoints) / configs.dataset_name / configs.dataset_id / configs.model_name / configs.model_id / f"{configs.seq_len}_{configs.pred_len}" / configs.subfolder_train / f"iter{configs.itr_i}" # same as the one in Exp_Main.train()
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Training iter{configs.itr_i} save to: {path}")
        with open(path / "configs.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(asdict(configs), f, default_flow_style=False)
        # init exp tracker
        if (configs.wandb and accelerator.is_main_process) or configs.sweep:
            import wandb
            run = wandb.init(
                # Set the project where this run will be logged
                project="YOUR_PROJECT_NAME",
                # Track hyperparameters and run metadata
                config={
                    "model_name": configs.model_name,
                    "model_id": configs.model_id,
                    "dataset_name": configs.dataset_name,
                    "seq_len": configs.seq_len,
                    "pred_len": configs.pred_len,
                    "learning_rate": configs.learning_rate,
                    "batch_size": configs.batch_size
                },
                dir=path
            )
            # overwrite model hyperparameters when sweeping
            for attribute_name in hyperparameters_sweep.keys():
                setattr(configs, attribute_name, getattr(wandb.config, attribute_name))

        accelerator.project_configuration.set_directories(project_dir=path)

        exp = Exp(configs)
        exp.train()
        return exp

    def train_with_auto_batch_reduction():
        """
        invoke start_exp_train(), and automatically reduce batch size on CUDA OOM errors.
        """
        while configs.batch_size >= 1:
            try:
                return start_exp_train()
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                if configs.batch_size == 1:
                    logger.exception("CUDA OOM error even with batch_size=1. Training aborted.")
                    exit(1)
                # Reduce batch size by half
                new_batch_size = max(1, configs.batch_size // 2)
                logger.error(f"CUDA OOM error! Reducing batch_size from {configs.batch_size} to {new_batch_size} and try again...")
                configs.batch_size = new_batch_size

    if configs.sweep:
        '''
        Currently, wandb sweep with huggingface accelerate multi GPU is tricky, use at your own risk.
        It is running N cases of hyperparameter settings at the same time, each case in a GPU. It's NOT running 1 case using all GPUs.
        - `wandb.sweep` is only created in the main process
        - `wandb.agent` is created in every process, where the sweep_id is obtained via tmp file on disk
        - `accelerate.backward` and `accelerate.log` are not used
        '''
        # hyperparameter search using wandb sweep
        logger.info('>>>>>>> sweeping start <<<<<<<')

        subfolder = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
        configs.subfolder_train = subfolder
        # Automatically enable wandb logging when sweeping
        configs.wandb = 1
        logger.debug('wandb=1: Weight & Bias logging is automatically enabled')

        # ignore itr, only train once for each combination
        configs.itr = 1
        configs.itr_i = 0
        logger.debug('itr=1: training iteration is automatically overwritten to 1')

        random.seed(fix_seed_list[configs.itr_i])
        torch.manual_seed(fix_seed_list[configs.itr_i])
        np.random.seed(fix_seed_list[configs.itr_i])

        exp = train_with_auto_batch_reduction()
        exp.test()
    elif configs.is_training:
        '''
        Normal train&test
        '''
        subfolder = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
        configs.subfolder_train = subfolder
        for i in range(configs.itr):
            configs.itr_i = i

            random.seed(fix_seed_list[i])
            torch.manual_seed(fix_seed_list[i])
            np.random.seed(fix_seed_list[i])

            exp = train_with_auto_batch_reduction()
            torch.cuda.empty_cache()
        exp.test()
    else:
        '''
        test only
        '''
        exp = Exp(configs)
        exp.test()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # warp the codes, such that errors will only be outputted from the main process
    try:
        if not configs.sweep:
            main()
        else:
            # first determine the hyperparameters actually accessed by model
            from utils.ExpConfigs import ExpConfigsTracker
            configs_tracker = ExpConfigsTracker(configs)
            model_module = importlib.import_module("models." + configs.model_name)
            model = model_module.Model(configs_tracker)
            del model
            accessed_configs: set[str] = configs_tracker.get_accessed_attributes()
            max_count = 1
            for accessed_config in accessed_configs:
                try:
                    ref_values = configs.get_sweep_values(accessed_config)
                    if ref_values and (type(ref_values) is list):
                        hyperparameters_sweep[accessed_config] = {
                            "values": ref_values
                        }
                        max_count *= len(ref_values)
                except Exception as e:
                    continue
            # grid search if <=16, otherwise bayes
            sweep_method = "grid" if max_count <= 16 else "bayes"
            max_count = min(max_count, 16)

            if hyperparameters_sweep == {}:
                logger.error(f"No hyperparameter to be searched, stopping now..")
                logger.debug(f"{configs.model_name} access these attributes in ExpConfigs:")
                configs_tracker.print_access_report()
                logger.debug("""Possible reasons: (1) The model does not access any hyperparameters in ExpConfigs; (2) The accessed hyperparameters have not set their metadata properly. Check the ExpConfigs class in utils/ExpConfigs.py. Example metadata setting:
                d_model: int = field(metadata={"sweep": [32, 64, 128, 256]})""")
                exit(0)
            else:
                logger.info(f"""{len(hyperparameters_sweep)} hyperparameters and {max_count} runs using "{sweep_method}" as the sweep method: \n{pprint.pformat(hyperparameters_sweep)}""")
                
            import wandb
            sweep_configuration = {
                "method": sweep_method,
                "metric": {"goal": "minimize", "name": "loss_val_best"},
                "parameters": hyperparameters_sweep
            }
            temp_file_path = "storage/tmp.txt"
            if accelerator.is_main_process:
                sweep_id = wandb.sweep(sweep=sweep_configuration, project="YOUR_PROJECT_NAME")
                with open(temp_file_path, mode='w', encoding="utf-8") as f:
                    f.write(sweep_id)
            accelerator.wait_for_everyone()
            sweep_id = None
            with open(temp_file_path, mode='r', encoding="utf-8") as f:
                sweep_id = f.readline()
            wandb.agent(
                sweep_id, 
                function=main, 
                project="YOUR_PROJECT_NAME",
                count=max_count
            )
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print("\nProcess interrupted...")
    except Exception as e:
        if accelerator.is_main_process:
            logger.exception(f"{e}", stack_info=True)
            exit(1)
