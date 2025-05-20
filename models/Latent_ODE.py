from einops import *
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal

from utils.globals import logger
from utils.ExpConfigs import ExpConfigs
import layers.Latent_ODE.utils as utils
from layers.Latent_ODE.ode_func import ODEFunc, ODEFunc_w_Poisson
from layers.Latent_ODE.diffeq_solver import DiffeqSolver
from layers.Latent_ODE.encoder_decoder import *
from layers.Latent_ODE.latent_ode import LatentODE

class Model(nn.Module):
    '''
    - paper: "Latent ODEs for Irregularly-Sampled Time Series" (NeurIPS 2019)
    - paper link: https://papers.nips.cc/paper_files/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html
    - Code adapted from: https://github.com/YuliaRubanova/latent_ode
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
            logger.exception("Latent ODE does not support classification task!", stack_info=True)
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
                "tp_to_predict": y_mark[0, :, 0],
                "observed_data": x,
                "observed_tp": x_mark[0, :, 0],
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

        dim = configs.d_model
        poisson = True if configs.task_name in ["short_term_forecast", "long_term_forecast"] else False # poisson=True is better when forecasting according to original paper
        if poisson:
            lambda_net = utils.create_net(dim, input_dim, 
                n_layers = 1, n_units = configs.latent_ode_units, nonlinear = nn.Tanh)

            # ODE function produces the gradient for latent state and for poisson rate
            ode_func_net = utils.create_net(dim * 2, configs.d_model * 2, 
                n_layers = configs.latent_ode_gen_layers, n_units = configs.latent_ode_units, nonlinear = nn.Tanh)

            gen_ode_func = ODEFunc_w_Poisson(
                input_dim = input_dim, 
                latent_dim = configs.d_model * 2,
                ode_func_net = ode_func_net,
                lambda_net = lambda_net,
                device = device)
        else:
            dim = configs.d_model 
            ode_func_net = utils.create_net(dim, configs.d_model, 
                n_layers = configs.latent_ode_gen_layers, n_units = configs.latent_ode_units, nonlinear = nn.Tanh)

            gen_ode_func = ODEFunc(
                input_dim = input_dim, 
                latent_dim = configs.d_model, 
                ode_func_net = ode_func_net,
                device = device)

        z0_diffeq_solver = None
        n_rec_dims = configs.latent_ode_rec_dims
        enc_input_dim = int(input_dim) * 2 # we concatenate the mask
        gen_data_dim = input_dim

        z0_dim = configs.d_model
        if poisson:
            z0_dim += configs.d_model # predict the initial poisson rate

        if configs.latent_ode_z0_encoder == "odernn":
            ode_func_net = utils.create_net(n_rec_dims, n_rec_dims, 
                n_layers = configs.latent_ode_rec_layers, n_units = configs.latent_ode_units, nonlinear = nn.Tanh)

            rec_ode_func = ODEFunc(
                input_dim = enc_input_dim, 
                latent_dim = n_rec_dims,
                ode_func_net = ode_func_net,
                device = device)

            z0_diffeq_solver = DiffeqSolver(enc_input_dim, rec_ode_func, "euler", configs.d_model, 
                odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
            
            encoder_z0 = Encoder_z0_ODE_RNN(n_rec_dims, enc_input_dim, z0_diffeq_solver, 
                z0_dim = z0_dim, n_gru_units = configs.latent_ode_gru_units, device = device)

        elif configs.latent_ode_z0_encoder == "rnn":
            encoder_z0 = Encoder_z0_RNN(z0_dim, enc_input_dim,
                lstm_output_size = n_rec_dims, device = device)
        else:
            raise Exception("Unknown encoder for Latent ODE model: " + configs.latent_ode_z0_encoder)

        decoder = Decoder(configs.d_model, gen_data_dim)

        diffeq_solver = DiffeqSolver(gen_data_dim, gen_ode_func, 'dopri5', configs.d_model, 
            odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

        model = LatentODE(
            input_dim = gen_data_dim, 
            latent_dim = configs.d_model, 
            encoder_z0 = encoder_z0, 
            decoder = decoder, 
            diffeq_solver = diffeq_solver, 
            z0_prior = z0_prior, 
            device = device,
            obsrv_std = obsrv_std,
            use_poisson_proc = poisson, 
            use_binary_classif = configs.latent_ode_classif,
            linear_classifier = configs.latent_ode_linear_classif,
            classif_per_tp = classif_per_tp,
            n_labels = n_labels,
            train_classif_w_reconstr = (configs.dataset_name == "P12")
        )

        return model