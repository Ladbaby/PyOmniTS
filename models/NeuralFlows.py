from einops import *
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal

from utils.globals import logger
from utils.ExpConfigs import ExpConfigs
from layers.NeuralFlows.experiments.latent_ode.lib.encoder_decoder import *
from layers.NeuralFlows.experiments.latent_ode.lib.latent_ode import LatentODE
from layers.NeuralFlows.models import CouplingFlow, ResNetFlow, GRUFlow

class Model(nn.Module):
    '''
    - paper: "Neural Flows: Efficient Alternative to Neural ODEs" (NeurIPS 2021)
    - paper link: https://proceedings.neurips.cc/paper/2021/hash/b21f9f98829dea9a48fd8aaddc1f159d-Abstract.html
    - Code adapted from: https://github.com/mbilos/neural-flows-experiments
    '''
    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.configs = configs
        self.pred_len = configs.pred_len_max_irr or configs.pred_len

        obsrv_std = 0.01
        if configs.dataset_name == "hopper":
            obsrv_std = 1e-3 

        obsrv_std = torch.Tensor([obsrv_std])

        z0_prior = Normal(torch.Tensor([0.0]), torch.Tensor([1.]))

        self.model = self.create_LatentODE_model(
            configs=configs, 
            input_dim=configs.enc_in, 
            z0_prior=z0_prior, 
            obsrv_std=obsrv_std, 
            device=configs.gpu_id, 
        )

        if configs.task_name == "classification":
            logger.exception("NeuralFlows does not support classification task!", stack_info=True)
            exit(1)

    def forward(
        self,
        x: Tensor, 
        x_mark: Tensor = None, 
        x_mask: Tensor = None, 
        y: Tensor = None, 
        y_mark: Tensor = None, 
        y_mask: Tensor = None,
        exp_stage: str = "train", 
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
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(torch.arange(end=y.shape[1], dtype=y.dtype, device=y.device) / y.shape[1], "L -> B L 1", B=y.shape[0])
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)

        # Encoder_z0_ODE_RNN minimum_step could be too small, or DiffeqSolver odeint raise "t must be strictly increasing or decreasing". Temporally reset timesteps
        # mark = torch.arange(x.shape[1] + y.shape[1], device=x.device, dtype=x.dtype) / (x.shape[1] + y.shape[1])
        # x_mark = mark[:x.shape[1]]
        # y_mark = mark[x.shape[1]: x.shape[1] + y.shape[1]]

        pred, loss = self.model(
            batch_dict={
                "tp_to_predict": y_mark[:, :, 0],
                "observed_data": x,
                "observed_tp": x_mark[:, :, 0],
                "observed_mask": x_mask,
                "data_to_predict": y,
                "mask_predicted_data": y_mask
            },
            n_traj_samples = 3,
            kl_coef = 0.
        )
        
        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast']:
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": torch.mean(pred, dim=0)[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
                "loss": loss
            }
        else:
            raise NotImplementedError

    def create_LatentODE_model(
        self, 
        configs: ExpConfigs, 
        input_dim, 
        z0_prior, 
        obsrv_std, 
        device, 
        classif_per_tp = False, 
        n_labels = 1
    ):
        classif_per_tp = (configs.dataset_name == 'HumanActivity')

        z0_diffeq_solver = None
        n_rec_dims = configs.latent_ode_rec_dims
        enc_input_dim = int(input_dim) * 2 # we concatenate the mask
        gen_data_dim = input_dim

        z0_dim = configs.neuralflows_latents
        hidden_dims = [configs.d_model] * configs.hidden_layers
        
        if configs.neuralflows_flow_model == 'coupling':
            flow = CouplingFlow
        elif configs.neuralflows_flow_model == 'resnet':
            flow = ResNetFlow
        elif configs.neuralflows_flow_model == 'gru':
            flow = GRUFlow
        else:
            raise ValueError('Unknown flow transformation')

        z0_diffeq_solver = SolverWrapper(flow(n_rec_dims, configs.neuralflows_flow_layers, hidden_dims, configs.neuralflows_time_net, configs.neuralflows_time_hidden_dim))
        diffeq_solver = SolverWrapper(flow(configs.neuralflows_latents, configs.neuralflows_flow_layers, hidden_dims, configs.neuralflows_time_net, configs.neuralflows_time_hidden_dim))

        encoder_z0 = Encoder_z0_ODE_RNN(
            n_rec_dims, 
            enc_input_dim, 
            z0_diffeq_solver, 
            z0_dim = z0_dim, 
            n_gru_units = configs.latent_ode_gru_units, 
            device = device
        )

        decoder = Decoder(configs.neuralflows_latents, gen_data_dim)

        return LatentODE(
            input_dim = gen_data_dim, 
            latent_dim = configs.d_model, 
            encoder_z0 = encoder_z0, 
            decoder = decoder, 
            diffeq_solver = diffeq_solver, 
            z0_prior = z0_prior, 
            device = device,
            obsrv_std = obsrv_std,
            use_poisson_proc = False, 
            use_binary_classif = configs.latent_ode_classif,
            linear_classifier = False,
            classif_per_tp = classif_per_tp,
            n_labels = n_labels,
            train_classif_w_reconstr = (configs.dataset_name in ["P12", "HumanActivity"])
        )

class SolverWrapper(nn.Module):
    def __init__(self, solver):
        super().__init__()
        self.solver = solver

    def forward(self, x, t, backwards=False):
        # print(f"{x.shape=}")
        # print(f"{t.shape=}")
        assert len(x.shape) - len(t.shape) == 1
        t = t.unsqueeze(-1)
        if t.shape[-3] != x.shape[-3]:
            t = t.repeat_interleave(x.shape[-3], dim=-3)
        if len(x.shape) == 4:
            t = t.repeat_interleave(x.shape[0], dim=0)
        y = self.solver(x, t) # (1, batch_size, times, dim)
        return y