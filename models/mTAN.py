import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from einops import repeat

from utils.globals import logger
from utils.ExpConfigs import ExpConfigs
from layers.mTAN.models import create_classifier, enc_mtan_rnn, dec_mtan_rnn

class Model(nn.Module):
    '''
    - paper: "Multi-Time Attention Networks for Irregularly Sampled Time Series" (ICLR 2021)
    - paper link: https://openreview.net/forum?id=4c0J6lwQ4_
    - code adapted from: https://github.com/reml-lab/mTAN
    '''
    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.configs = configs
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.latent_dim = 16
        rec_hidden = 64 # random.choice([32, 64, 128])
        gen_hidden = 50
        num_ref_points = configs.mtan_num_ref_points # random.choice([8, 16, 32, 64, 128])
        self.k_iwae = 8
        self.alpha = configs.mtan_alpha

        self.enc = enc_mtan_rnn(
            input_dim=configs.enc_in,
            query=torch.linspace(0, 1., num_ref_points),
            latent_dim=self.latent_dim,
            nhidden=rec_hidden,
            embed_time=128,
            learn_emb=True,
            num_heads=configs.n_heads
        )
        self.dec = dec_mtan_rnn(
            input_dim=configs.dec_in,
            query=torch.linspace(0, 1., num_ref_points),
            latent_dim=self.latent_dim,
            nhidden=gen_hidden,
            embed_time=128,
            learn_emb=True,
            num_heads=configs.n_heads,
        )

        if configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
            self.reconstruction_loss_fn = ReconstructionLoss()
        elif configs.task_name == "classification":
            self.classifier = create_classifier(self.latent_dim, rec_hidden)
            self.reconstruction_loss_fn = ReconstructionLoss()
            self.classification_loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor = None, 
        x_mask: Tensor = None, 
        y: Tensor = None, 
        y_mark: Tensor = None, 
        y_mask: Tensor = None, 
        y_class: Tensor = None,
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
        if y_class is None:
            if self.configs.task_name == "classification":
                logger.warning(f"y_class is missing for the model input. This is only reasonable when the model is testing flops!")
            y_class = torch.ones((BATCH_SIZE), dtype=x.dtype, device=x.device)

        batch_len = x.shape[0]
        subsampled_tp = x_mark[:, :, 0]
        if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            observed_tp = y_mark[:, :, 0]
        elif self.configs.task_name in ["classification", "imputation"]:
            observed_tp = x_mark[:, :, 0]
        else:
            raise NotImplementedError
        # END adaptor


        out = self.enc(torch.cat((x, x_mask), 2), subsampled_tp)
        qz0_mean = out[:, :, :self.latent_dim]
        qz0_logvar = out[:, :, self.latent_dim:]
        epsilon = torch.randn(
            self.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
        ).to(x.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2]) # (BATCH_SIZE * k_iwae, num_ref_points, latent_dim)
        pred_x = self.dec(
            z0,
            observed_tp[None, :, :].repeat(self.k_iwae, 1, 1).view(-1, observed_tp.shape[1])
        )

        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast', "imputation"]:
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            if exp_stage in ["train", "val"]:
                # we have to expand the batch_size dimension to batch_size * k_iwae when training. true and mask should also be expanded to align with the common api's shape
                loss = self.reconstruction_loss_fn(
                    pred=pred_x,
                    true=repeat(y, "b l f -> (n b) l f", n=self.k_iwae),
                    mask=repeat(y_mask, "b l f -> (n b) l f", n=self.k_iwae),
                    qz0_mean=qz0_mean,
                    qz0_logvar=qz0_logvar
                )
                return {
                    "loss": loss
                }
            else:
                return {
                    "pred": pred_x.view(self.k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2]).mean(0)[:, -PRED_LEN:, f_dim:],
                    "true": y[:, :, f_dim:],
                    "mask": y_mask[:, :, f_dim:],
                }
        elif self.configs.task_name == "classification":
            pred_y = self.classifier(z0)
            if exp_stage in ["train", "val"]:
                recon_loss = self.reconstruction_loss_fn(
                    pred=pred_x,
                    true=repeat(x, "b l f -> (n b) l f", n=self.k_iwae),
                    mask=repeat(x_mask, "b l f -> (n b) l f", n=self.k_iwae),
                    qz0_mean=qz0_mean,
                    qz0_logvar=qz0_logvar
                )
                ce_loss = self.classification_loss_fn(pred_y, y_class.unsqueeze(0).repeat_interleave(self.k_iwae, 0).view(-1))
                loss = recon_loss + self.alpha * ce_loss
                return {
                    "pred_class": pred_y,
                    "true": y_class,
                    "loss": loss
                }
            else:
                return {
                    "pred_class": pred_y.view(self.k_iwae, batch_len, pred_y.shape[1]).mean(0),
                    "true": y_class
                }
        else:
            raise NotImplementedError

class ReconstructionLoss(nn.Module):
    '''
    modified VAE training loss for mTAN
    https://openreview.net/forum?id=4c0J6lwQ4_
    '''
    def __init__(self, std=0.01, norm=True, k_iwae=8, kl_coef=1):
        self.std = std
        self.norm = norm
        self.k_iwae = k_iwae
        self.kl_coef = kl_coef
        super(ReconstructionLoss, self).__init__()

    def forward(
        self,
        pred,
        true,
        qz0_mean,
        qz0_logvar,
        current_epoch=0,
        mask=None,
        **kwargs
    ):
        # BEGIN adaptor
        # WARNING: unlike other loss functions, this loss assumes the input to have size batch_size * k_iwae in the first dimension
        pred = pred.view(self.k_iwae, int(pred.shape[0] / self.k_iwae), pred.shape[1], pred.shape[2])
        true = true[:int(true.shape[0] / self.k_iwae)]
        mask = mask[:int(mask.shape[0] / self.k_iwae)] if mask is not None else torch.ones_like(true, device=true.device)
        # END adaptor

        wait_until_kl_inc = 10
        if current_epoch < wait_until_kl_inc:
            self.kl_coef = 0.
        else:
            self.kl_coef = (1 - 0.99 ** (current_epoch - wait_until_kl_inc))

        noise_std = self.std  # default 0.1
        noise_std_ = torch.zeros(pred.size()).to(pred.device) + noise_std
        noise_logvar = 2. * torch.log(noise_std_).to(pred.device)
        logpx = self.log_normal_pdf(true, pred, noise_logvar,
                            mask).sum(-1).sum(-1)
        pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(pred.device)
        analytic_kl = self.normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1).sum(-1)
        if self.norm:
            logpx /= mask.sum(-1).sum(-1) + 1e-8
            analytic_kl /= mask.sum(-1).sum(-1) + 1e-8

        loss = -(torch.logsumexp(logpx - self.kl_coef * analytic_kl, dim=0).mean(0) - np.log(self.k_iwae))

        return loss

    def log_normal_pdf(self, x, mean, logvar, mask):
        const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
        const = torch.log(const)
        return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask

    
    def normal_kl(self, mu1, lv1, mu2, lv2):
        v1 = torch.exp(lv1)
        v2 = torch.exp(lv2)
        lstd1 = lv1 / 2.
        lstd2 = lv2 / 2.

        kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
        return kl
    