from einops import *
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal

from utils.globals import logger
from utils.ExpConfigs import ExpConfigs
from layers.GNeuralFlow.experiments.latent_ode.lib.encoder_decoder import *
from layers.GNeuralFlow.experiments.latent_ode.lib.latent_ode import LatentODE
from layers.GNeuralFlow.models import CouplingFlow_latent, ResNetFlow, GRUFlow

class Model(nn.Module):
    '''
    - paper: "Graph Neural Flows for Unveiling Systemic Interactions Among Irregularly Sampled Time Series" (NeurIPS 2024)
    - paper link: https://openreview.net/forum?id=tFB5SsabVb
    - Code adapted from: https://github.com/gmerca/GNeuralFlow
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
            logger.exception("GNeuralFlow does not support classification task!", stack_info=True)
            exit(1)

        # Initialize graph learner
        self.graph_learner = GraphLearner(configs.enc_in)

    def forward(
        self,
        x: Tensor, 
        x_mark: Tensor = None, 
        x_mask: Tensor=None, 
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

        # We have to dynamically construct graph for input time series
        # Forward pass to get adjacency matrix
        adj_matrix = self.graph_learner().to(x.device)
        # END adaptor

        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast', "imputation"]:
            pred, loss = self.model(
                batch_dict={
                    "tp_to_predict": y_mark[:, :, 0],
                    "observed_data": x,
                    "observed_tp": x_mark[:, :, 0],
                    "observed_mask": x_mask,
                    "data_to_predict": y,
                    "mask_predicted_data": y_mask,
                    "adj": adj_matrix
                },
                n_traj_samples = 3,
                kl_coef = 0.
            )
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
            flow = CouplingFlow_latent
        elif configs.neuralflows_flow_model == 'resnet':
            flow = ResNetFlow
        elif configs.neuralflows_flow_model == 'gru':
            flow = GRUFlow
        else:
            raise ValueError('Unknown flow transformation')

        z0_diffeq_solver = SolverWrapper(flow(
            dim=n_rec_dims, 
            n_layers=configs.neuralflows_flow_layers, 
            hidden_dims=hidden_dims, 
            time_net=configs.neuralflows_time_net, 
            time_hidden_dim=configs.neuralflows_time_hidden_dim
        ))
        diffeq_solver = SolverWrapper(flow(configs.neuralflows_latents, configs.neuralflows_flow_layers, hidden_dims, configs.neuralflows_time_net, configs.neuralflows_time_hidden_dim))

        encoder_z0 = Encoder_z0_ODE_RNN(
            latent_dim = n_rec_dims, 
            input_dim = enc_input_dim, 
            z0_diffeq_solver = z0_diffeq_solver, 
            z0_dim = z0_dim, 
            n_gru_units = configs.latent_ode_gru_units, 
            device = device,
            nfeats = configs.enc_in,
            nsens = 1,
            dim = 1,
            enc_type = "rnn2",
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

    def forward(self, x, t, h=None):
        assert len(x.shape) - len(t.shape) == 1
        t = t.unsqueeze(-1)
        if t.shape[-3] != x.shape[-3]:
            t = t.repeat_interleave(x.shape[-3], dim=-3)
        if len(x.shape) == 4:
            t = t.repeat_interleave(x.shape[0], dim=0)
        y = self.solver(x, h, t)  # (1, batch_size, times, dim)
        return y

class GraphLearner(nn.Module):
    def __init__(self, n_variables):
        super(GraphLearner, self).__init__()
        self.adjacency_matrix = nn.Parameter(torch.randn(n_variables, n_variables)).cuda()

    def forward(self):
        # Optionally apply some transformation
        adj_matrix = F.softmax(self.adjacency_matrix, dim=-1)  # Row-wise softmax
        return adj_matrix