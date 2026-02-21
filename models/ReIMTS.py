# Code from: https://github.com/Ladbaby/PyOmniTS
import importlib
import math
import subprocess
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from einops import rearrange, repeat
from torch import Tensor

from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class Model(nn.Module):
    '''
    - paper: "Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting" (ICLR 2026)
    - paper link: https://openreview.net/forum?id=JEIDxiTWzB
    - code adapted from: https://github.com/Ladbaby/PyOmniTS
    '''
    def __init__(
        self, 
        configs: ExpConfigs,
        current_level: int = 0
    ) -> None:
        super(Model, self).__init__()
        logger.info(f"---Creating scale level {current_level}---")
        self.task_name = configs.task_name
        self.configs = configs
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.seq_len = configs.seq_len_max_irr or configs.seq_len # equal to seq_len_max_irr if not None, else seq_len
        self.patch_len = configs.patch_len_max_irr or configs.patch_len # equal to seq_len_max_irr if not None, else seq_len
        self.current_level = current_level

        assert configs.ts_backbone_name is not None, "Please specify the name of time series backbone for model ReIMTS. e.g., --ts_backbone_name GraFITi"

        if configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            self.time_len_list = [configs.seq_len + configs.pred_len] + configs.patch_len_list
            self.n_levels = len(self.time_len_list)
            if configs.seq_len_max_irr is not None:
                self.time_len_max_irr_list = [configs.seq_len_max_irr + configs.pred_len_max_irr]
                for i in range(1, self.n_levels):
                    self.time_len_max_irr_list.append(math.ceil(self.time_len_max_irr_list[i - 1] / math.ceil(self.time_len_list[i - 1] / self.time_len_list[i])))
        elif configs.task_name == "classification":
            self.time_len_list = [configs.seq_len] + configs.patch_len_list
            self.n_levels = len(self.time_len_list)
            self.time_len_max_irr_list = [configs.seq_len]
            for i in range(1, self.n_levels):
                self.time_len_max_irr_list.append(math.ceil(self.time_len_max_irr_list[i - 1] / math.ceil(self.time_len_list[i - 1] / self.time_len_list[i])))
        else:
            raise NotImplementedError()
        self.current_time_len = self.time_len_list[self.current_level]
        if self.current_level == 0:
            self.n_patch_all = 1
        else:
            self.n_patch_all = math.ceil(self.time_len_list[self.current_level - 1] / self.time_len_list[self.current_level])
        
        # load default config of backbone model
        yaml_configs_path = f"configs/{configs.ts_backbone_name}/{configs.ts_backbone_name}/{configs.dataset_name}.yaml" # It assumes backbone's --model_id the same as --model_name
        if not Path(yaml_configs_path).exists():
            # a tricky way to generate backbone's default config
            cmd = ["sh", f"scripts/{configs.ts_backbone_name}/{configs.dataset_name}.sh"]
            logger.info(f"Running '{' '.join(cmd)}' for 10 seconds to generate 'configs/{configs.ts_backbone_name}/{configs.ts_backbone_name}/{configs.dataset_name}.yaml'")
            proc = subprocess.Popen(cmd)
            try:
                proc.wait(timeout=10)   # wait up to 10 seconds
            except subprocess.TimeoutExpired:
                proc.kill()             # force kill after timeout
                proc.wait()             # reap the process
            if Path(yaml_configs_path).exists():
                logger.info(f"Successfully generate {yaml_configs_path}")
            else:
                logger.exception(f"Failed to automatically generate yaml configuration file '{yaml_configs_path}'. Please try to run '{' '.join(cmd)}' manually.")
                exit(0)
        with open(yaml_configs_path, 'r', encoding="utf-8") as stream:
            try:
                yaml_configs: dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.exception(f"Exception when parsing {yaml_configs_path}: {exc}", stack_info=True)
                exit(1)
        if yaml_configs is not None:
            unknown_key_list: list[str] = []
            for key in yaml_configs.keys():
                if key not in configs.__dict__.keys():
                    logger.debug(f"Remove unknown key {key} from backbone's configs")
                    unknown_key_list.append(key)
            for key in unknown_key_list:
                yaml_configs.pop(key)
            try:
                configs_backbone = ExpConfigs(**yaml_configs)
            except Exception as e:
                logger.debug(e)
                logger.debug("Trying to directly use argparse result instead...")
                configs_backbone = Namespace(**yaml_configs)
        else:
            logger.exception(f"{yaml_configs=}")
            exit(1)

        # overwrite part of the backbone config
        ts_backbone_overwrite_config_list = configs.ts_backbone_overwrite_config_list + ["pred_len", "pred_len_max_irr", "task_name", "n_classes"] # additionally overwrite these args
        for config_name in ts_backbone_overwrite_config_list:
            # user-defined list of argument that values from adaptor class will overwrite backbones'
            assert hasattr(configs_backbone, config_name), f"Argument '{config_name}' is not found in config file '{yaml_configs_path}'. Make sure the argument name provided in --backbone_overwrite_config_list is correct!"
            logger.debug(f"Config '{config_name}' in backbone {configs.ts_backbone_name} is overwritten: {getattr(configs_backbone, config_name)} -> {getattr(configs, config_name)}")
            setattr(configs_backbone, config_name, getattr(configs, config_name))

        configs_backbone.batch_size = configs.batch_size
        if configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            configs_backbone.seq_len = configs.seq_len + configs.pred_len
            if configs.seq_len_max_irr is not None:
                configs_backbone.seq_len_max_irr = configs.seq_len_max_irr + configs.pred_len_max_irr
        elif configs.task_name == "classification":
            configs_backbone.seq_len = configs.seq_len
            if configs.seq_len_max_irr is not None:
                configs_backbone.seq_len_max_irr = configs.seq_len_max_irr
        else:
            raise NotImplementedError()
        for i in range(self.current_level + 1):
            if i == 0:
                continue
            else:
                configs_backbone.batch_size *= math.ceil(self.time_len_list[i - 1] / self.time_len_list[i])
                configs_backbone.seq_len = math.ceil(configs_backbone.seq_len / math.ceil(self.time_len_list[i - 1] / self.time_len_list[i]))
                if configs.seq_len_max_irr is not None:
                    configs_backbone.seq_len_max_irr = math.ceil(configs_backbone.seq_len_max_irr / math.ceil(self.time_len_list[i - 1] / self.time_len_list[i]))

        logger.debug(f"""ReIMTS also overwrites the following configs in backbone {configs.ts_backbone_name}:
        - batch_size: -> {configs_backbone.batch_size}
        - seq_len: -> {configs_backbone.seq_len}
        - seq_len_max_irr: -> {configs_backbone.seq_len_max_irr}
        - pred_len: -> {configs_backbone.pred_len}
        - pred_len_max_irr: -> {configs_backbone.pred_len_max_irr}
        - task_name: -> {configs_backbone.task_name}
        - n_classes: -> {configs_backbone.n_classes}""")


        # dynamic backbone model import & construction
        backbone_module = importlib.import_module("layers.ReIMTS.models." + configs.ts_backbone_name)
        self.backbone = backbone_module.Model(
            configs=configs_backbone,
            current_level=current_level,
            time_len_list=self.time_len_list
        )

        if self.current_level < self.n_levels - 1:
            # recursively creates model in each scale level
            self.next_model = Model(
                configs=configs,
                current_level=current_level + 1
            )

    def forward(
        self, 
        x: Tensor, 
        x_mark: Tensor | None = None, 
        x_mask: Tensor | None = None, 
        x_repr_time: Tensor | None = None,
        x_repr_var: Tensor | None = None,
        x_repr_obs: Tensor | None = None,
        y: Tensor | None = None,
        y_mark: Tensor | None = None, 
        y_mask: Tensor | None = None,
        y_class: Tensor | None = None,
        exp_stage: str = "train",
        train_stage: int = 1,
        **kwargs
    ):
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.pred_len
        if x_mark is None:
            x_mark = repeat(torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device) / x.shape[1], "L -> B L 1", B=x.shape[0])
        if x_mask is None:
            x_mask = torch.ones_like(x, device=x.device, dtype=x.dtype)
        if y is None:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(torch.arange(end=y.shape[1], dtype=y.dtype, device=y.device) / y.shape[1], "L -> B L 1", B=y.shape[0])
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)
        if y_class is None:
            if self.configs.task_name == "classification":
                logger.warning(f"y_class is missing for the model input. This is only reasonable when the model is testing flops!")
            y_class = torch.ones((BATCH_SIZE), dtype=x.dtype, device=x.device)
        # END adaptor

        if self.current_level == 0:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
                x_padding = torch.zeros_like(y)
                x = torch.cat([x, x_padding], dim=1)
                x_mask = torch.cat([x_mask, x_padding], dim=1)
                x_mark = torch.cat([x_mark, y_mark], dim=1)

        input_dict = {
            "x": x,
            "x_mark": x_mark,
            "x_mask": x_mask,
            "x_repr_time": x_repr_time,
            "x_repr_var": x_repr_var,
            "x_repr_obs": x_repr_obs,
            "y": y,
            "y_mark": y_mark,
            "y_mask": y_mask,
            "y_class": y_class,
            "exp_stage": exp_stage,
            "train_stage": train_stage
        }

        backbone_output: dict[Tensor] = self.backbone(**input_dict)

        if self.current_level == self.n_levels - 1:
            # base case in lowest layer, recursion stops.
            return backbone_output
        else:
            # (B, L, ENC_IN) -> (B, L_NEW, (ENC_IN * PATCH_LEN))
            x_patched = self.patchify(x)
            x_mark_patched = self.patchify(x_mark)
            x_mask_patched = self.patchify(x_mask)

            N_PATCH_ALL_NEXT_FRACTAL = math.ceil(self.time_len_list[self.current_level] / self.time_len_list[self.current_level + 1])
            input_dict = {
                "x": x_patched,
                "x_mark": x_mark_patched,
                "x_mask": x_mask_patched,
                "exp_stage": exp_stage,
                "train_stage": train_stage
            }

            # Add task specific inputs
            if self.task_name == "classification":
                input_dict["y_class"] = self.patchify_classification(y_class)
            elif self.task_name in ["short_term_forecast", "long_term_forecast"]:
                input_dict["y"] = torch.repeat_interleave(y, N_PATCH_ALL_NEXT_FRACTAL, dim=0)
                input_dict["y_mark"] = torch.repeat_interleave(y_mark, N_PATCH_ALL_NEXT_FRACTAL, dim=0)
                input_dict["y_mask"] = torch.repeat_interleave(y_mask, N_PATCH_ALL_NEXT_FRACTAL, dim=0)
            else:
                raise NotImplementedError()

            # Add model specific inputs
            if "pred_repr_time" in backbone_output.keys():
                # temporal representation case
                input_dict["x_repr_time"] = self.patchify_repr_time(backbone_output["pred_repr_time"])
            if "pred_repr_var" in backbone_output.keys():
                # variable representation case
                input_dict["x_repr_var"] = torch.repeat_interleave(backbone_output["pred_repr_var"], N_PATCH_ALL_NEXT_FRACTAL, dim=0)
            if "pred_repr_obs" in backbone_output.keys():
                # observational representation case
                input_dict["x_repr_obs"] = self.patchify_repr_time(backbone_output["pred_repr_obs"])
            
            next_model_output: dict[Tensor] = self.next_model(**input_dict) # recursively invoke model in next scale level
            if exp_stage in ["train", "val"]:
                return next_model_output
            elif exp_stage == "test":
                if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
                    if next_model_output["pred"].shape[0] > BATCH_SIZE:
                        next_model_output["pred"] = self.unpatchify(next_model_output["pred"])
                    if self.current_level == 0:
                        next_model_output["pred"] = next_model_output["pred"][:, -self.pred_len:]
                elif self.configs.task_name == "classification":
                    if next_model_output["pred_class"].shape[0] > BATCH_SIZE:
                        next_model_output["pred_class"] = self.unpatchify_classification(next_model_output["pred_class"])
                    if "pred_repr_var" in next_model_output.keys():
                        if next_model_output["pred_repr_var"].shape[0] > BATCH_SIZE:
                            next_model_output["pred_repr_var"] = self.unpatchify_repr_var(next_model_output["pred_repr_var"])
                    if "pred_repr_time" in next_model_output.keys():
                        if next_model_output["pred_repr_time"].shape[0] > BATCH_SIZE:
                            next_model_output["pred_repr_time"] = self.unpatchify(next_model_output["pred_repr_time"])
                else:
                    raise NotImplementedError()
                return next_model_output

    def patchify(self, x):
        N_PATCH_ALL_NEXT_FRACTAL = math.ceil(self.time_len_list[self.current_level] / self.time_len_list[self.current_level + 1])
        padding_len = N_PATCH_ALL_NEXT_FRACTAL * self.time_len_max_irr_list[self.current_level + 1] - x.shape[1]
        if padding_len > 0:
            x_padding = torch.zeros(x.shape[0], padding_len, x.shape[2]).to(x.device)
            x = torch.cat([x, x_padding], dim=1)
        return rearrange(x, "B (N_PATCH PATCH_LEN) ENC_IN -> (B N_PATCH) PATCH_LEN ENC_IN", N_PATCH=N_PATCH_ALL_NEXT_FRACTAL)

    def patchify_classification(self, x):
        N_PATCH_ALL_NEXT_FRACTAL = math.ceil(self.time_len_list[self.current_level] / self.time_len_list[self.current_level + 1])
        return repeat(x, "B N_CLASSES -> (B N_PATCH) N_CLASSES", N_PATCH=N_PATCH_ALL_NEXT_FRACTAL)

    def patchify_repr_time(self, x):
        N_PATCH_ALL_NEXT_FRACTAL = math.ceil(self.time_len_list[self.current_level] / self.time_len_list[self.current_level + 1])
        if self.configs.reimts_pad_time_emb:
            padding_len = N_PATCH_ALL_NEXT_FRACTAL * self.time_len_max_irr_list[self.current_level + 1] - x.shape[1]
            if padding_len > 0:
                x_padding = torch.zeros(x.shape[0], padding_len, x.shape[2]).to(x.device)
                x = torch.cat([x, x_padding], dim=1)
        return rearrange(x, "B (N_PATCH PATCH_LEN) D -> (B N_PATCH) PATCH_LEN D", N_PATCH=N_PATCH_ALL_NEXT_FRACTAL)

    def unpatchify(self, x):
        N_PATCH_ALL_NEXT_FRACTAL = math.ceil(self.time_len_list[self.current_level] / self.time_len_list[self.current_level + 1])
        return rearrange(x, "(B N_PATCH) PATCH_LEN ENC_IN -> B (N_PATCH PATCH_LEN) ENC_IN", N_PATCH=N_PATCH_ALL_NEXT_FRACTAL)

    def unpatchify_classification(self, x):
        N_PATCH_ALL_NEXT_FRACTAL = math.ceil(self.time_len_list[self.current_level] / self.time_len_list[self.current_level + 1])
        return rearrange(x, "(B N_PATCH) N_CLASSES -> B N_PATCH N_CLASSES", N_PATCH=N_PATCH_ALL_NEXT_FRACTAL).mean(1)

    def unpatchify_repr_var(self, x):
        N_PATCH_ALL_NEXT_FRACTAL = math.ceil(self.time_len_list[self.current_level] / self.time_len_list[self.current_level + 1])
        return rearrange(x, "(B N_PATCH) C_OUT D -> B C_OUT (N_PATCH D)", N_PATCH=N_PATCH_ALL_NEXT_FRACTAL)


