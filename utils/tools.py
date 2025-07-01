import time
import json
import socket
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.globals import logger, accelerator

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if val_loss in [np.nan, torch.nan, float("nan")]:
            logger.warning(f"Validation loss is nan, stopping...")
            self.early_stop = True
            return

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.debug(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            logger.debug(f'Validation loss decreased ({self.val_loss_min:.2e} --> {val_loss:.2e}).  Saving model ...')
        accelerator.save_model(
            model, 
            path, 
            safe_serialization=False
        )
        self.val_loss_min = val_loss

def test_params_flop(
    model: torch.nn.Module,
    x_shape: tuple[int],
    model_id: str,
    task_key: str
):
    """
    you need to give default value to all arguments in model.forward(), the following code can only pass the first argument `x` to forward()

    - task_key: forecasting, etc...
    """
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model.eval().cuda(), 
            x_shape, 
            as_strings=False, 
            print_per_layer_stat=False
        )
        logger.info(f"{model_id} with input shape {x_shape}")
        logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))

        SEQ_LEN, ENC_IN = x_shape

        input_config = f"seq_len_{SEQ_LEN}_enc_in_{ENC_IN}"

        complexity = {
            "macs": macs,
            "params": params
        }

        if Path(f"metrics/{task_key}/model_complexities.json").exists():
            with open(f"metrics/{task_key}/model_complexities.json", "r") as f:
                complexities = json.load(f)

            if input_config not in complexities.keys():
                complexities[input_config] = {
                    model_id: complexity
                }
            else:
                if model_id not in complexities[input_config].keys():
                    complexities[input_config][model_id] = complexity
                else:
                    if complexities[input_config][model_id] != complexity:
                        logger.warning(f"""
                        Existing model complexity in metrics/{task_key}/model_complexities.json for input shape {x_shape} and model {model_id} is not the same as newly measured data.

                        Existing data: {complexities[input_config][model_id]}
                        Newly measured data: {complexity}
                        
                        model_complexities.json will preserve the existing data.
                        """)
        else:
            complexities = {
                input_config: {
                    model_id: complexity
                }
            }

        with open(f"metrics/{task_key}/model_complexities.json", "w") as f:
            json.dump(complexities, f, indent=2)
            logger.info(f"metrics/{task_key}/model_complexities.json saved.")

def test_train_time(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    model_id: str,
    dataset_name: str,
    gpu: int,
    seq_len: int,
    pred_len: int,
    task_key: str,
    retain_graph: int
):
    '''
    test model's time consumption for 1 forward and 1 backward, in ms
    '''
    model = model.train()

    time_start = time.time() * 1000
    for batch in tqdm(dataloader):
        batch = {k: v.float().to(f"cuda:{gpu}") for k, v in batch.items()}
        outputs = model(
            exp_stage="train",
            **batch
        )
        loss = criterion(
            **outputs
        )["loss"]
        loss.backward(retain_graph=retain_graph)
    torch.cuda.current_stream().synchronize()
    time_end = time.time() * 1000
    train_time_mean = (time_end - time_start) / len(dataloader)

    logger.info(f"{model_id} with {seq_len=} and {pred_len=}")
    logger.info(f"{train_time_mean=:.2f}")

    input_config = f"{seq_len}/{pred_len}"
    host_name = socket.gethostname()

    if Path(f"metrics/{task_key}/model_train_time.json").exists():
        with open(f"metrics/{task_key}/model_train_time.json", "r") as f:
            train_time_dict: dict = json.load(f)

        train_time_dict.setdefault(host_name, {}).setdefault(dataset_name, {}).setdefault(input_config, {}).setdefault(model_id, None)

        if train_time_dict[host_name][dataset_name][input_config][model_id] not in [train_time_mean, None]:
            logger.warning(f"""
            Existing model inference speed in metrics/{task_key}/model_train_time.json on host {host_name} for seq_len/pred_len {seq_len}/{pred_len} and model {model_id} is not the same as newly measured data.

            Existing data: {train_time_dict[host_name][dataset_name][input_config][model_id]}
            Newly measured data: {train_time_mean}
            
            model_train_time.json will preserve the existing data.
            """)
        else:
            train_time_dict[host_name][dataset_name][input_config][model_id] = train_time_mean
    else:
        train_time_dict = {
            host_name: {
                dataset_name: {
                    input_config: {
                        model_id: train_time_mean
                    }
                }
            }
        }

    with open(f"metrics/{task_key}/model_train_time.json", "w") as f:
        json.dump(train_time_dict, f, indent=2)
        logger.info(f"metrics/{task_key}/model_train_time.json saved.")

def test_gpu_memory(
    model: torch.nn.Module,
    batch: dict[torch.Tensor],
    model_id: str,
    dataset_name: str,
    gpu: int,
    seq_len: int,
    pred_len: int,
    task_key: str
):
    '''
    gpu memory usage at model's training time (without pytorch's cuda driver and runtime)
    '''
    torch.cuda.reset_peak_memory_stats()
    model(
        exp_stage="train",
        **batch
    )
    peak_memory = torch.cuda.max_memory_allocated(gpu) / (1024 ** 3)
    logger.info(f"Peak GPU memory usage for {model_id}: {peak_memory} GB")

    input_config = f"{seq_len}/{pred_len}"
    if Path(f"metrics/{task_key}/model_gpu_memories.json").exists():
        with open(f"metrics/{task_key}/model_gpu_memories.json", "r") as f:
            gpu_memory_dict = json.load(f)

        gpu_memory_dict.setdefault(dataset_name, {}).setdefault(input_config, {}).setdefault(model_id, None)

        if gpu_memory_dict[dataset_name][input_config][model_id] not in [peak_memory, None]:
            logger.warning(f"""
            Existing model gpu memory usage in metrics/{task_key}/model_gpu_memories.json for input shape {batch["x"].shape} and model {model_id} is not the same as newly measured data.

            Existing data: {gpu_memory_dict[dataset_name][input_config][model_id]}
            Newly measured data: {peak_memory}
            
            model_gpu_memories.json will preserve the existing data.
            """)
        else:
            gpu_memory_dict[dataset_name][input_config][model_id] = peak_memory
    else:
        gpu_memory_dict = {
            dataset_name: {
                input_config: {
                    model_id: peak_memory
                }
            }
        }

    with open(f"metrics/{task_key}/model_gpu_memories.json", "w") as f:
        json.dump(gpu_memory_dict, f, indent=2)
        logger.info(f"metrics/{task_key}/model_gpu_memories.json saved.")

def linear_interpolation(x):
    # Linear interpolation function
    # Assuming x is a tensor of shape (batch_size, sequence_length, input_size)
    # Interpolate n-1 values between n original values
    batch_size, time_length, n_variables = x.shape
    x_interpolated = torch.zeros(batch_size, 2 * time_length - 1, n_variables, device=x.device)
    x_interpolated[:, 0] = x[:, 0]
    interpolated_values = (x[:, 1:] + x[:, :-1]) / 2
    # for i in range(batch_size):
    for j in range(time_length - 1):
        x_interpolated[:, 2 * j + 1] = interpolated_values[:, j]
        x_interpolated[:, 2 * j] = x[:, j]

    return x_interpolated