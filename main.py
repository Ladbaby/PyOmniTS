import random
import datetime

import torch
import numpy as np

from exp.exp_main import Exp_Main
from utils.globals import logger, accelerator
from utils.configs import configs

def main():
    # random seed
    fix_seed_list = range(2024, 2024 + configs.itr)

    configs.use_gpu = True if torch.cuda.is_available() and configs.use_gpu else False

    Exp = Exp_Main

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

        exp = Exp(configs)

        exp.train()
        exp.test()

    elif configs.is_training:
        subfolder = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
        configs.subfolder_train = subfolder
        for i in range(configs.itr):
            configs.itr_i = i

            random.seed(fix_seed_list[i])
            torch.manual_seed(fix_seed_list[i])
            np.random.seed(fix_seed_list[i])

            exp = Exp(configs)
            exp.train()

            torch.cuda.empty_cache()
        exp.test()
    else:
        exp = Exp(configs)
        exp.test()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # warp the codes, such that errors will only be outputted from the main process
    try:
        if not configs.sweep:
            main()
        else:
            import wandb
            sweep_configuration = {
                "method": "grid",
                "metric": {"goal": "minimize", "name": "loss_val"},
                "parameters": {
                    "learning_rate": {"values": [0.01, 0.001, 0.0001, 0.00001]},
                    "batch_size": {"values": [16, 32, 64, 128]},
                },
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
                count=16
            )
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print("\nProcess interrupted...")
    except Exception as e:
        if accelerator.is_main_process:
            logger.exception(f"{e}", stack_info=True)
            exit(1)
