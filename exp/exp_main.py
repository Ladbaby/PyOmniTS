import urllib.request
import os
from pathlib import Path
import datetime
import warnings
import yaml
import json
from collections import OrderedDict
from typing import Generator
from dataclasses import asdict
import importlib

import numpy as np
from tqdm import tqdm
import torch
from torch import optim, Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, LRScheduler
from torchvision.datasets.utils import download_url
from accelerate import load_checkpoint_in_model

from data.data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, test_params_flop, test_train_time, test_gpu_memory
from utils.metrics import metric, metric_classification
from utils.globals import logger, accelerator
from utils.ExpConfigs import ExpConfigs

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, configs: ExpConfigs):
        super(Exp_Main, self).__init__(configs)

    def _build_model(self) -> Module:
        # build the adaptor class
        model_module = importlib.import_module("models._OpenMIC_Adaptor") # WARNING: strictly assumes the outer model is _OpenMIC_Adaptor 
        model = model_module.Model(self.configs)
        return model

    def _get_data(self, flag: str) -> tuple[Dataset, DataLoader]:
        data_set, data_loader = data_provider(self.configs, flag)
        return data_set, data_loader

    def _select_optimizer(self, model: Module) -> optim.Optimizer:
        model_optim = optim.Adam(model.parameters(), lr=self.configs.learning_rate)
        return model_optim

    def _select_lr_scheduler(self, optimizer: optim.Optimizer) -> LRScheduler:
        # Initialize scheduler based on configs.lradj
        if self.configs.lr_scheduler == 'ExponentialDecayLR':
            '''
            Originally named as 'type1'
            '''
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** epoch)
        elif self.configs.lr_scheduler == 'ManualMilestonesLR':
            '''
            Originally named as 'type2'
            '''
            from lr_schedulers.ManualMilestonesLR import ManualMilestonesLR
            # Convert 1-based epochs to 0-based
            user_milestones = {2:5e-5, 4:1e-5, 6:5e-6, 8:1e-6, 10:5e-7, 15:1e-7, 20:5e-8}
            milestones = {k-1: v for k, v in user_milestones.items()}
            scheduler = ManualMilestonesLR(optimizer, milestones)
        elif self.configs.lr_scheduler == 'DelayedStepDecayLR':
            '''
            Originally named as 'type3'
            '''
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 if epoch < 2 else (0.8 ** (epoch - 2)))
        elif self.configs.lr_scheduler == 'CosineAnnealingLR':
            '''
            Originally named as 'cosine'
            '''
            scheduler = CosineAnnealingLR(optimizer, T_max=self.configs.train_epochs, eta_min=0.0)
        elif self.configs.lr_scheduler == "MultiStepLR":
            '''
            Configured following CSDI
            '''
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[0.75 * self.configs.train_epochs, 0.9 * self.configs.train_epochs], gamma=0.1
            )
        else:
            logger.exception(f"Unknown lr scheduler '{self.configs.lr_scheduler}'", stack_info=True)
            exit(1)

        return scheduler

    def _select_criterion(self) -> Module:
        # dynamically import the desired loss function
        loss_module = importlib.import_module("loss_fns." + self.configs.loss)
        criterion = loss_module.Loss()
        return criterion

    def _get_state_dict(self, path: Path) -> OrderedDict:
        '''
        Fix model state dict errors
        '''
        logger.info(f"Loading model checkpoint from {path}")
        state_dict = torch.load(path)
        new_state_dict = OrderedDict()
        if_fixed = False
        for key, value in state_dict.items():
            # you may insert modifications to the key and value here
            if 's4' in key and (('B' in key or 'P' in key or 'w' in key) and ('weight' not in key)):
                # S4 layer don't need to load these weights
                if_fixed = True
                continue
            new_state_dict[f"backbone.{key}"] = value.contiguous() # WARNING: strictly assumes the outer model is _OpenMIC_Adaptor 
        if if_fixed:
            logger.warning("Automatically fixed model state dict errors. It may cause unexpected behavior!")
        return new_state_dict

    def _check_model_outputs(self, batch:dict, outputs:dict) -> None:
        '''
        Perform necessary checks on model's outputs
        '''
        # check if the data type is dict
        if type(outputs) is not dict:
            logger.exception(f"Expect model's forward function to return dict. Current output's data type is {type(outputs)}.", stack_info=True)
            exit(1)

        if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            # check if outputs' true is the the same as input dataset's y
            if "true" in outputs.keys() and not torch.equal(batch["y"], outputs["true"]):
                logger.warning(f"Model's outputs['true'] is not equal to input's batch['y']. Please confirm that you are not using input's batch['y'] as ground truth. This is expected in some models such as diffusion.")
        elif self.configs.task_name in ["classification"]:
            if "true_class" in outputs.keys():
                # check if outputs' true_class is the the same as input dataset's y_class
                if not torch.equal(batch["y_class"], outputs["true_class"]):
                    logger.warning(f"Model's outputs['true_class'] is not equal to input's batch['y_class']. Please confirm that you are not using input's batch['y_class'] as ground truth.")
                # check data type and LongTensor dtype for CrossEntropyLoss:
                # if outputs["true_class"].dtype is not torch.int64:
                #     logger.warning(f"batch['true_class'] is expected to have dtype torch.int64. Currently it has dtype {outputs['true_class'].dtype}")

    def _merge_gathered_dicts(self, dicts: list[dict]) -> dict:
        '''
        manually merge list of dictionary gathered when testing
        accelerate.gather_for_metrics may have unexpected behavior, thus merge manually instead
        '''
        merged_dict = {}
        keys_not_returned = []
        for d in dicts:
            for key, tensor in d.items():
                if type(tensor).__name__ != "Tensor":
                    # skip value that is not Pytorch Tensor
                    if key not in keys_not_returned:
                        keys_not_returned.append(key)
                        logger.warning(f"{key=} will not be gathered for metric calculation in test, since its value has data type '{type(tensor).__name__}', which is not 'Tensor'")
                    continue
                if key in merged_dict:
                    merged_dict[key] = torch.cat((merged_dict[key], tensor.detach().cpu()), dim=0)
                else:
                    merged_dict[key] = tensor.detach().cpu()
        return merged_dict

    def vali(
        self, 
        model_train: Module, 
        vali_loader: DataLoader, 
        criterion: Module, 
        current_epoch: int
    ) -> np.ndarray:
        total_loss = []
        model_train.eval()
        with torch.no_grad():
            with tqdm(total=len(vali_loader), leave=False, desc="Validating") as it:
                batch: dict[str, Tensor] # type hints
                for i, batch in enumerate(vali_loader):
                    # warn if the size does not match
                    if batch[next(iter(batch))].shape[0] != self.configs.batch_size and current_epoch == 0:
                        logger.warning(f"Batch No.{i} of total {len(vali_loader)} has actual batch_size={batch[next(iter(batch))].shape[0]}, which is not the same as --batch_size={self.configs.batch_size}")
                    if "y_mask" in batch.keys():
                        if torch.sum(batch["y_mask"]).item() == 0:
                            if current_epoch == 0:
                                logger.warning(f"Batch No.{i} of total {len(vali_loader)} has no evaluation point (inferred from y_mask), thus skipping")
                            continue
                    if not self.configs.use_multi_gpu:
                        batch = {k: v.to(f"cuda:{self.configs.gpu_id}") for k, v in batch.items()}

                    # some model's forward function return different values in "train", "val", "test", they can use `exp_stage` as argument to distinguish
                    outputs: dict[str, Tensor] = model_train(
                        exp_stage="val",
                        **batch
                    )

                    loss: Tensor = criterion(
                        **outputs
                    )["loss"]
                    total_loss.append(loss.item())

                    if accelerator.is_main_process:
                        # update only in main process
                        it.update()
                        it.set_postfix(loss=f"{loss.item():.2e}")
        total_loss = np.average(total_loss)
        model_train.train()
        return total_loss

    def train(self) -> None:
        logger.info('>>>>>>> training start <<<<<<<')
        # save training config file for reference
        path = Path(self.configs.checkpoints) / self.configs.dataset_name / self.configs.model_name / self.configs.subfolder_train / f"iter{self.configs.itr_i}"
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Training iter{self.configs.itr_i} save to: {path}")
        with open(path / "configs.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self.configs), f, default_flow_style=False)

        accelerator.project_configuration.set_directories(project_dir=path)

        # init exp tracker
        if (self.configs.wandb and accelerator.is_main_process) or self.configs.sweep:
            import wandb
            run = wandb.init(
                # Set the project where this run will be logged
                project="YOUR_PROJECT_NAME",
                # Track hyperparameters and run metadata
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
            if self.configs.sweep:
                # overwrite default configs by wandb.config when sweeping
                self.configs.learning_rate = wandb.config.learning_rate
                self.configs.batch_size = wandb.config.batch_size

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # model initialized after dataset to obtain possible dynamic information from dataset (e.g., seq_len_max_irr)
        model_train = self._build_model()

        model_optim = self._select_optimizer(model_train)
        lr_scheduler = self._select_lr_scheduler(model_optim)
        criterion = self._select_criterion()

        if not self.configs.sweep:
            train_loader, vali_loader, model_train, model_optim = accelerator.prepare(
                train_loader, vali_loader, model_train, model_optim
            )
            accelerator.register_for_checkpointing(model_optim)
        else:
            model_train, model_optim = accelerator.prepare(
                model_train, model_optim
            )

        # Save initial states
        initial_optimizer_state = model_optim.state_dict()
        initial_scheduler_state = lr_scheduler.state_dict()

        if not self.configs.use_multi_gpu:
            model_train = model_train.to(f"cuda:{self.configs.gpu_id}")

        for train_stage in range(1, self.configs.n_train_stages + 1):
            early_stopping = EarlyStopping(patience=self.configs.patience, verbose=True)
            logger.info(f"Train stage {train_stage}/{self.configs.n_train_stages} starts.")
            for epoch in tqdm(range(self.configs.train_epochs), desc="Epochs"):
                train_loss = []
                model_train.train()
                with tqdm(total=len(train_loader), leave=False, desc="Training") as it:
                    batch: dict[str, Tensor] # type hints
                    for i, batch in enumerate(train_loader):
                        # warn if the size does not match
                        if batch[next(iter(batch))].shape[0] != self.configs.batch_size and epoch == 0:
                            logger.warning(f"Batch No.{i} of total {len(train_loader)} has actual batch_size={batch[next(iter(batch))].shape[0]}, which is not the same as --batch_size={self.configs.batch_size}")
                        if "y_mask" in batch.keys():
                            if torch.sum(batch["y_mask"]).item() == 0:
                                if epoch == 0:
                                    logger.warning(f"Batch No.{i} of total {len(train_loader)} has no evaluation point (inferred from y_mask), thus skipping")
                                continue
                        model_optim.zero_grad()
                        if not self.configs.use_multi_gpu:
                            batch = {k: v.to(f"cuda:{self.configs.gpu_id}") for k, v in batch.items()}

                        outputs: dict[str, Tensor] = model_train(
                            exp_stage="train",
                            train_stage=train_stage,
                            **batch
                        )

                        # check model's outputs only in the first iteration
                        if i == 0 and epoch == 0:
                            self._check_model_outputs(batch, outputs)
                        
                        loss: Tensor = criterion(
                            current_epoch=epoch,
                            **outputs
                        )["loss"]

                        # check loss
                        if torch.any(torch.isnan(loss)):
                            logger.exception("Loss is nan! Training interruptted!")
                            for key, value in outputs.items():
                                if key == "loss":
                                    continue
                                elif type(value).__name__ != "Tensor" and torch.any(torch.isnan(value)):
                                    logger.error(f"Nan value found in model's output tensor '{key}' of shape {value.shape}: {value}")
                            logger.info("Hint: possible cause for nan loss: 1. large learning rate; 2. sqrt(0); 3. ReLU->LeakyReLU")
                            exit(1)

                        train_loss.append(loss.item())

                        if accelerator.is_main_process:
                            # update progress bar only in main process
                            it.update()
                            it.set_postfix(loss=f"{loss.item():.2e}")

                        if self.configs.sweep:
                            loss.backward(retain_graph=self.configs.retain_graph)
                        else:
                            accelerator.backward(loss, retain_graph=self.configs.retain_graph)
                        if self.configs.task_name == "classification":
                            clip_grad_norm_(model_train.parameters(), max_norm=4.0)
                        model_optim.step()

                # DEBUG: state saving is disabled to minimize disk write time
                # save the state of optimizer
                # if not self.configs.sweep:
                #     accelerator.save_state(safe_serialization=False)

                # validation
                if epoch % self.configs.val_interval == 0:
                    vali_loss = self.vali(model_train, vali_loader, criterion, epoch)
                    if (self.configs.wandb and accelerator.is_main_process) or self.configs.sweep:
                        wandb.log({
                            "loss_train": np.mean(train_loss),
                            "loss_val": vali_loss
                        })
                    early_stopping(vali_loss, model_train, path)
                    if early_stopping.early_stop:
                        logger.info("Early stopping")
                        accelerator.set_trigger()
                elif (self.configs.wandb and accelerator.is_main_process) or self.configs.sweep:
                    wandb.log({
                        "loss_train": np.mean(train_loss),
                    })

                lr_scheduler.step()
                logger.debug(f'Updating learning rate to {lr_scheduler.get_last_lr()[0]:.6e}')
                if accelerator.check_trigger():
                    accelerator.wait_for_everyone()
                    break

            # Reset optimizer, scheduler
            model_optim.load_state_dict(initial_optimizer_state)
            lr_scheduler.load_state_dict(initial_scheduler_state)


    def test(self) -> None:
        logger.info('>>>>>>> testing start <<<<<<<')

        # convert task_name to task_key for storage folder naming
        task_key_mapping = {
            "short_term_forecast": "forecasting",
            "long_term_forecast": "forecasting",
        }
        if self.configs.test_flop:
            self.configs.batch_size = 1
            logger.debug("batch_size automatically overwritten to 1.")
            test_params_flop(
                model=self._build_model().to(self.device), 
                x_shape=(self.configs.seq_len,self.configs.enc_in),
                model_name=self.configs.model_name,
                task_key=task_key_mapping[self.configs.task_name] if self.configs.task_name in task_key_mapping.keys() else self.configs.task_name
            )
            exit(0)

        if self.configs.test_train_time:
            self.configs.batch_size = 32
            logger.debug("batch_size automatically overwritten to 32.")
            train_data, train_loader = self._get_data(flag='train')
            test_train_time(
                model=self._build_model().to(self.device), 
                dataloader=train_loader,
                criterion=self._select_criterion(),
                model_name=self.configs.model_name,
                dataset_name=self.configs.dataset_name,
                gpu=self.configs.gpu_id,
                seq_len=self.configs.seq_len,
                pred_len=self.configs.pred_len,
                task_key=task_key_mapping[self.configs.task_name] if self.configs.task_name in task_key_mapping.keys() else self.configs.task_name,
                retain_graph=self.configs.retain_graph
            )
            exit(0)

        if self.configs.test_gpu_memory:
            self.configs.batch_size = 32
            logger.debug("batch_size automatically overwritten to 32.")
            train_data, train_loader = self._get_data(flag='train')
            batch = next(iter(train_loader))
            batch = {k: v.to(f"cuda:{self.configs.gpu_id}") for k, v in batch.items()}
            model = self._build_model().to(self.device).train()
            test_gpu_memory(
                model=model,
                batch=batch,
                model_name=self.configs.model_name,
                dataset_name=self.configs.dataset_name,
                gpu=self.configs.gpu_id,
                seq_len=self.configs.seq_len,
                pred_len=self.configs.pred_len,
                task_key=task_key_mapping[self.configs.task_name] if self.configs.task_name in task_key_mapping.keys() else self.configs.task_name
            )
            exit(0)

        if self.configs.test_dataset_statistics:
            _, data_loader = self._get_data(flag='test_all')
            n_observations_raw = 0
            n_observations_all = 0
            logger.info(f"""Testing Dataset '{self.configs.dataset_name}':
            - seq_len={self.configs.seq_len}
            - pred_len={self.configs.pred_len}
            - batch_size={self.configs.batch_size}
            - collate_fn='{self.configs.collate_fn}'""")
            logger.warning("Make sure seq_len and pred_len are correctly set.")
            for batch in tqdm(data_loader):
                n_observations_raw += np.sum(batch["x_mask"].detach().cpu().numpy())
                n_observations_raw += np.sum(batch["y_mask"].detach().cpu().numpy())
                n_observations_all += np.sum(np.ones_like(batch["x_mask"].detach().cpu().numpy()))
                n_observations_all += np.sum(np.ones_like(batch["y_mask"].detach().cpu().numpy()))

            logger.info(f"No. observations (raw): {n_observations_raw}")
            logger.info(f"No. observations (all): {n_observations_all}")
            exit(0)

        # test_all will test the model on all available sets (train, val, test). Needs to be supported by the dataset
        flag = "test_all" if self.configs.test_all else "test"
        test_data, test_loader = self._get_data(flag=flag)

        # find model checkpoint path
        checkpoint_location: Path = None
        actual_itrs = 1
        if self.configs.checkpoints_test is None:
            # by default, if checkpoints_test is not given, it tries to load the latest corresponding checkpoint
            checkpoint_location = Path(self.configs.checkpoints) / self.configs.dataset_name / self.configs.model_name
            if self.configs.load_checkpoints_test:
                try:
                    # first, find the latest one based on timestamp in name
                    child_folders = [(entry.name.replace(f"{self.configs.model_id}_", ""), entry) for entry in checkpoint_location.iterdir() if entry.is_dir() and self.configs.model_id in entry.name]
                    if len(child_folders) == 0:
                        logger.exception(f"No folder under '{checkpoint_location}' matches the model_id '{self.configs.model_id}'.", stack_info=True)
                        logger.exception(f"Tips: Failed to infer the latest checkpoint folder. Please manually provide the checkpoints_test argument pointing to the folder of checkpoint file")
                        exit(1)
                    latest_folder: str = sorted(child_folders, key=lambda item: datetime.datetime.strptime(item[0], "%m%d_%H%M"))[-1][1].name
                    checkpoint_location = checkpoint_location / latest_folder
                    # then find the latest iter
                    actual_itrs = len([entry.name for entry in checkpoint_location.iterdir() if entry.is_dir()])
                except Exception as e:
                    logger.exception(f"{e}", stack_info=True)
                    logger.exception(f"Tips: Failed to infer the latest checkpoint folder. Please manually provide the checkpoints_test argument pointing to the folder of checkpoint file")
                    exit(1)
            else:
                # create pseudo training directory for test results
                train_folder = f'{self.configs.model_id}_{datetime.datetime.now().strftime("%m%d_%H%M")}'
                path = checkpoint_location / train_folder / f"iter0"
                path.mkdir(parents=True, exist_ok=True)
                checkpoint_location = checkpoint_location / train_folder


        # test on all iters' checkpoints
        for itr_i in range(actual_itrs):
            if self.configs.checkpoints_test is None:
                checkpoint_location_itr = checkpoint_location / f"iter{itr_i}"
            else:
                checkpoint_location_itr = Path(self.configs.checkpoints_test)

            model_test = self._build_model().eval()
            # load model checkpoint if load_checkpoints_test
            if self.configs.load_checkpoints_test:
                checkpoint_file = checkpoint_location_itr / "pytorch_model.bin"
                if checkpoint_file.exists():
                    try: 
                        # model state dict cannot be modified after accelerator.prepare
                        original_state_dict = self._get_state_dict(checkpoint_file)
                        load_result = model_test.load_state_dict(original_state_dict, strict=False)
                        if load_result.missing_keys or load_result.unexpected_keys:
                            missing_keys = []
                            for key in load_result.missing_keys:
                                if not key.startswith("vggish."):
                                    missing_keys.append(key)
                            if len(missing_keys) > 0:
                                logger.warning(f"Missing keys in model weights: {missing_keys}")
                            if load_result.unexpected_keys:
                                logger.warning(f"Unexpected keys in checkpoint file: {load_result.unexpected_keys}")
                            if len(missing_keys) > 0 or load_result.unexpected_keys:
                                logger.warning("Results may be incorrect!")

                    except Exception as e:
                        logger.exception(f"{e}", stack_info=True)
                        logger.exception(f"Failed to load checkpoint file at {checkpoint_file}. Skipping it...")
                        continue
                else:
                    try:
                        # when weights are large (>10GB), they will be saved in several files
                        load_checkpoint_in_model(model_test, checkpoint_location_itr)
                    except Exception as e:
                        logger.exception(f"{e}", stack_info=True)
                        logger.exception(f"Failed to load checkpoint file at {checkpoint_file}. Skipping it...")
                        continue

            model_test, test_loader = accelerator.prepare(model_test, test_loader)
            if not self.configs.use_multi_gpu:
                model_test = model_test.to(f"cuda:{self.configs.gpu_id}")

            # create folder for test results
            subfolder_eval = f'eval_{datetime.datetime.now().strftime("%m%d_%H%M")}'
            folder_path = checkpoint_location_itr / subfolder_eval
            folder_path.mkdir(exist_ok=True)
            logger.info(f"Testing results will be saved under {folder_path}")

            # dictionary holding input and output data
            array_dict = {}
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
                input_tensor_names = ["x", "y", "x_mask", "y_mask", "sample_ID"]
                output_tensor_names = ["pred"]
            elif self.configs.task_name in ["classification"]:
                # input_tensor_names = ["x", "y_class", "x_mask", "y_mask", "sample_ID"]
                input_tensor_names = ["y_class"]
                output_tensor_names = ["pred_class"]
            elif self.configs.task_name in ["representation_learning"]:
                input_tensor_names = []
                # output_tensor_names = ["pred_repr_time", "pred_repr_var", "pred_repr_obs"] # pred_repr_obs is super large that can case OOM
                output_tensor_names = ["pred_repr_time"]

            for tensor_name in input_tensor_names + output_tensor_names:
                array_dict[tensor_name] = []
            
            with torch.no_grad():
                batch: dict[str, Tensor] # type hints
                for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), leave=False, desc="Testing"):
                    # warn if the size does not match
                    if batch[next(iter(batch))].shape[0] != self.configs.batch_size:
                        logger.warning(f"Batch No.{i} of total {len(test_loader)} has actual batch_size={batch[next(iter(batch))].shape[0]}, which is not the same as --batch_size={self.configs.batch_size}")
                        continue
                    if not self.configs.use_multi_gpu:
                        batch = {k: v.to(f"cuda:{self.configs.gpu_id}") for k, v in batch.items()}

                    outputs: dict[str, Tensor] = model_test(
                        exp_stage="test",
                        **batch
                    )

                    # check model's outputs only in the first iteration
                    if i == 0 and itr_i == 0:
                        self._check_model_outputs(batch, outputs)

                    batch_all: list[dict] = accelerator.gather_for_metrics([batch])
                    batch_all: dict = self._merge_gathered_dicts(batch_all)
                    outputs_all: list[dict] = accelerator.gather_for_metrics([outputs])
                    outputs_all: dict = self._merge_gathered_dicts(outputs_all)

                    for tensor_name in input_tensor_names:
                        if tensor_name in batch_all.keys():
                            array_dict[tensor_name].append(batch_all[tensor_name].detach().cpu().numpy())
                    for tensor_name in output_tensor_names:
                        if tensor_name in outputs_all.keys():
                            array_dict[tensor_name].append(outputs_all[tensor_name].detach().cpu().numpy())

            for tensor_name in input_tensor_names + output_tensor_names:
                if tensor_name in array_dict.keys():
                    array_dict[tensor_name] = np.concatenate(array_dict[tensor_name], axis=0)

            metrics = None
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
                metrics = metric(**array_dict)
                if (self.configs.wandb and accelerator.is_main_process and self.configs.is_training) or self.configs.sweep:
                    import wandb
                    wandb.log({
                        "loss_test": np.mean(metrics["MSE"]),
                    })
            elif self.configs.task_name in ["classification"]:
                self.configs: ExpConfigs # type hints
                metrics = metric_classification(
                    **array_dict,
                    n_classes=self.configs.n_classes
                )
            
            if metrics is not None:
                # convert to float before saving to json
                for key, value in metrics.items():
                    if isinstance(value, np.float32):
                        metrics[key] = float(value)
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, np.float32):
                                metrics[key] = [float(v) for v in value]
                                break
                logger.info("Metrics:\n%s", json.dumps(metrics, indent=4)) # log result in a readable way
                with open(folder_path / "metric.json", "w") as f:
                    json.dump(metrics, f, indent=2)

            if self.configs.save_arrays:
                for tensor_name in input_tensor_names:
                    np.save(folder_path / f"input_{tensor_name}.npy", array_dict[tensor_name])
                for tensor_name in output_tensor_names:
                    np.save(folder_path / f"output_{tensor_name}.npy", array_dict[tensor_name])

    def inference(self, inference_loader: DataLoader = None) -> Generator[dict[str, Tensor], None, None]:
        '''
        Similar to test(), but only obtain the output of model

        - inference_loader: overwrite default behavior in special cases
        '''
        logger.info('>>>>>>> inference start <<<<<<<')

        if inference_loader is None:
            flag = "test_all" if self.configs.test_all else "test"
            _, inference_loader = self._get_data(flag=flag)

        # find model checkpoint path is removed
        if self.configs.checkpoints_test is None:
            logger.exception(f"inference() requires --checkpoints_test in arguments!", stack_info=True)
            exit(1)

        checkpoint_location_itr = Path(self.configs.checkpoints_test)

        model_inference = self._build_model().to(self.device).eval()
        # load model checkpoint if load_checkpoints_test
        if self.configs.load_checkpoints_test:
            checkpoint_file = checkpoint_location_itr / "pytorch_model.bin"
            # WARNING: the following logic assumes weights are < 10GB
            if not checkpoint_file.exists():
                # try downloading from web 
                url = f"https://huggingface.co/Ladbaby/InsRec-models/resolve/main/{self.configs.dataset_name}/{self.configs.model_name}/pytorch_model.bin?download=true"
                download_choice = input(f'''The checkpoint file for model '{self.configs.model_name}' on dataset '{self.configs.dataset_name}' is going to be downloaded at '{checkpoint_file}' via url {url}, proceed? (Y/N):''')
                while True:
                    if download_choice.upper() == 'Y':
                        break
                    elif download_choice.upper() == 'N':
                        logger.info("Download aborted.")
                        exit(0)
                    else:
                        download_choice = input(f"Invalid choice '{download_choice}', please select between Y and N:")

                # Read proxy settings from environment variables (both lowercase and uppercase)
                http_proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
                https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')

                proxies = {}
                if http_proxy:
                    proxies['http'] = http_proxy
                    proxies['https'] = http_proxy # if https_proxy is not present, use http_proxy for https traffic instead
                if https_proxy:
                    proxies['https'] = https_proxy

                # Install proxy handler if proxies are found
                if proxies:
                    logger.debug(f"Using proxy read from environments for download: {proxies}")
                    proxy_handler = urllib.request.ProxyHandler(proxies)
                    opener = urllib.request.build_opener(proxy_handler)
                    urllib.request.install_opener(opener)
                try:
                    download_url(
                        url=url,
                        root=checkpoint_location_itr,
                        filename="pytorch_model.bin"
                    )
                except Exception as e:
                    logger.exception(e, stack_info=True)
            try: 
                # model state dict cannot be modified after accelerator.prepare
                load_result = model_inference.load_state_dict(self._get_state_dict(checkpoint_file), strict=False)
                if load_result.missing_keys or load_result.unexpected_keys:
                    missing_keys = []
                    for key in load_result.missing_keys:
                        if not key.startswith("vggish."):
                            missing_keys.append(key)
                    if len(missing_keys) > 0:
                        logger.warning(f"Missing keys in model weights: {missing_keys}")
                    if load_result.unexpected_keys:
                        logger.warning(f"Unexpected keys in checkpoint file: {load_result.unexpected_keys}")
                    if len(missing_keys) > 0 or load_result.unexpected_keys:
                        logger.warning("Results may be incorrect!")
            except Exception as e:
                logger.exception(f"{e}", stack_info=True)
                logger.exception(f"Failed to load checkpoint file at '{checkpoint_file}'.")
                logger.warning(f"It is possible that the checkpoint file at '{checkpoint_file}' is broken. Try manually remove it then rerun.")
                

        model_inference, inference_loader = accelerator.prepare(model_inference, inference_loader)
        if not self.configs.use_multi_gpu:
            model_inference = model_inference.to(f"cuda:{self.configs.gpu_id}")

        with torch.no_grad():
            batch: dict[str, Tensor] # type hints
            for i, batch in tqdm(enumerate(inference_loader), total=len(inference_loader), leave=False, desc="Testing"):
                # warn if the size does not match
                if batch[next(iter(batch))].shape[0] != self.configs.batch_size:
                    logger.warning(f"Batch No.{i} of total {len(inference_loader)} has actual batch_size={batch[next(iter(batch))].shape[0]}, which is not the same as --batch_size={self.configs.batch_size}")
                    continue
                if not self.configs.use_multi_gpu:
                    batch = {k: v.to(f"cuda:{self.configs.gpu_id}") for k, v in batch.items()}

                outputs: dict[str, Tensor] = model_inference(
                    exp_stage="test",
                    **batch
                )

                batch_all: list[dict] = accelerator.gather_for_metrics([batch])
                batch_all: dict = self._merge_gathered_dicts(batch_all)
                outputs_all: list[dict] = accelerator.gather_for_metrics([outputs])
                outputs_all: dict = self._merge_gathered_dicts(outputs_all)

                yield batch_all, outputs_all