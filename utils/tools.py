# Code from: https://github.com/Ladbaby/PyOmniTS
import hashlib
import json
import os
import signal
import socket
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import requests
import torch
from requests.adapters import HTTPAdapter
from torch.utils.data import DataLoader
from tqdm import tqdm
from urllib3.util.retry import Retry

from utils.globals import accelerator, logger


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
    logger.warning(f"Reminder: replace '*' with torch.mul, '@' with torch.matmul to get the most accurate result.")
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model.eval().cuda(), 
            x_shape, 
            as_strings=False, 
            print_per_layer_stat=True,
            verbose=True
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
                        overwrite_choice = input(f"""Existing model complexity in metrics/{task_key}/model_complexities.json for input shape {x_shape} and model {model_id} is not the same as newly measured data.

Existing data: {complexities[input_config][model_id]}
Newly measured data: {complexity}

Do you want to overwrite the existing data? (Y/N)""")
                        while True:
                            if overwrite_choice.upper() == 'Y':
                                complexities[input_config][model_id] = complexity
                                logger.info("Newly measured data saved.")
                                break
                            elif overwrite_choice.upper() == 'N':
                                logger.info("model_complexities.json will preserve the existing data.")
                                exit(0)
                            else:
                                overwrite_choice = input(f"Invalid choice '{overwrite_choice}', please select between Y and N:")
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

def download_file(
    url: str, 
    local_file_path: Path | str,
    max_retries: int = 3,
    chunk_size: int = 8192,
    expected_checksum: Optional[str] = None,
    checksum_algorithm: Literal['md5', 'sha1', 'sha256', 'sha512'] = 'sha256'
):
    """
    Download a file with optional checksum validation.
    
    Args:
        url: URL to download from
        local_file_path: Path where file should be saved
        max_retries: Maximum number of retry attempts
        chunk_size: Size of chunks to download at a time
        expected_checksum: Expected checksum value for validation (optional)
        checksum_algorithm: Hashing algorithm to use ('md5', 'sha1', 'sha256', 'sha512')
    """
    if type(local_file_path) is str:
        local_file_path = Path(local_file_path)
    
    # Check if file exists and validate checksum if provided
    if local_file_path.exists() and expected_checksum:
        logger.info("File exists, verifying checksum...")
        if verify_checksum(local_file_path, expected_checksum, checksum_algorithm):
            logger.info("File already exists and checksum matches. Skipping download.")
            return
        else:
            logger.warning("File exists but checksum doesn't match. Re-downloading...")
            local_file_path.unlink()
    elif local_file_path.exists():
        logger.info("File already exists and no checksum provided. Skipping download.")
        return
    
    def timeout_handler(signum, frame):
        raise TimeoutError()

    if not local_file_path.exists():
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            download_choice = input(f"{local_file_path} not found. Do you want to download it from '{url}'? (y/n, default y): ")
            signal.alarm(0)  # Cancel the alarm
        except TimeoutError:
            print("\nNo response received. Defaulting to 'y'.")
            download_choice = 'y'

        # Default to 'y' if user just pressed Enter
        if download_choice.strip() == '':
            download_choice = 'y'
        
        while True:
            if download_choice.lower() == 'y':
                break
            elif download_choice.lower() == 'n':
                logger.info("Download aborted.")
                exit(0)
            else:
                download_choice = input(f"Invalid choice '{download_choice}', please select between y and n: ")
                if download_choice.strip() == '':
                    download_choice = 'y'
        
        # Read proxy settings from environment variables
        proxies = {}
        http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
        if http_proxy:
            proxies['http'] = http_proxy
        
        https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        if https_proxy:
            proxies['https'] = https_proxy
        
        logger.debug(f"Using proxy configs: {proxies}")
        
        # Create session with retry strategy
        session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Headers to improve compatibility
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Encoding': 'identity',  # Disable compression
            'Connection': 'keep-alive'
        }
        
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"Download attempt {attempt + 1}")
                
                # Make request with extended timeouts
                response = session.get(
                    url, 
                    stream=True, 
                    proxies=proxies if proxies else None,
                    headers=headers,
                    timeout=(30, 300)  # (connect_timeout, read_timeout)
                )
                response.raise_for_status()
                
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                total_size = int(response.headers.get('content-length', 0))
                logger.debug(f"Expected file size: {total_size} bytes")
                
                downloaded_size = 0
                
                # Initialize hash object if checksum validation is requested
                hash_obj = None
                if expected_checksum:
                    hash_obj = hashlib.new(checksum_algorithm)
                
                with open(local_file_path, 'wb') as file:
                    with tqdm(
                        desc=f"Downloading {local_file_path.name}",
                        total=total_size if total_size > 0 else None,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        dynamic_ncols=True
                    ) as bar:
                        try:
                            for chunk in response.iter_content(
                                chunk_size=chunk_size, 
                                decode_unicode=False
                            ):
                                if chunk:  # Filter out keep-alive chunks
                                    file.write(chunk)
                                    downloaded_size += len(chunk)
                                    bar.update(len(chunk))
                                    
                                    # Update hash if checksum validation is enabled
                                    if hash_obj:
                                        hash_obj.update(chunk)

                        except Exception as e:
                            logger.warning(f"Error during download: {e}")
                            raise
                
                # Verify download completeness
                actual_size = local_file_path.stat().st_size
                logger.info(f"Download completed: {actual_size} bytes")
                
                if total_size > 0 and actual_size != total_size:
                    logger.warning(f"Size mismatch: expected {total_size}, got {actual_size}")
                    if attempt < max_retries:
                        logger.info("Retrying download...")
                        local_file_path.unlink()  # Remove incomplete file
                        time.sleep(2)  # Wait before retry
                        continue
                
                # Validate checksum if provided
                if expected_checksum:
                    calculated_checksum = hash_obj.hexdigest().lower()
                    expected_lower = expected_checksum.lower()
                    
                    logger.info(f"Verifying {checksum_algorithm} checksum...")
                    
                    if calculated_checksum == expected_lower:
                        logger.info("Checksum verification passed.")
                    else:
                        logger.error(f"Checksum verification failed!")
                        logger.error(f"Expected {checksum_algorithm}: {expected_lower}")
                        logger.error(f"Calculated {checksum_algorithm}: {calculated_checksum}")
                        
                        if attempt < max_retries:
                            logger.info("Retrying download due to checksum mismatch...")
                            local_file_path.unlink()  # Remove corrupted file
                            time.sleep(2)  # Wait before retry
                            continue
                        else:
                            # Remove corrupted file and raise error
                            local_file_path.unlink()
                            raise ValueError(f"Checksum verification failed after {max_retries + 1} attempts")
                
                logger.info("Download finished successfully.")
                break
                
            except (requests.exceptions.RequestException, IOError) as e:
                logger.error(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying in 2 seconds... ({max_retries - attempt} attempts left)")
                    if local_file_path.exists():
                        local_file_path.unlink()  # Remove incomplete file
                    time.sleep(2)
                else:
                    logger.error("All download attempts failed.")
                    raise
        
        session.close()

def verify_checksum(
    file_path: Path, 
    expected_checksum: str, 
    algorithm: str = 'sha256'
) -> bool:
    """
    Verify the checksum of a file.
    
    Args:
        file_path: Path to the file to verify
        expected_checksum: Expected checksum value
        algorithm: Hashing algorithm to use
        
    Returns:
        True if checksum matches, False otherwise
    """
    try:
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        
        calculated_checksum = hash_obj.hexdigest().lower()
        expected_lower = expected_checksum.lower()
        
        return calculated_checksum == expected_lower
        
    except Exception as e:
        logger.error(f"Error calculating checksum: {e}")
        return False
