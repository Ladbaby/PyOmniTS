import torch

import layers.GNeuralFlow.experiments.latent_ode.lib.utils as utils
from layers.GNeuralFlow.experiments.latent_ode.lib.encoder_decoder import *
from layers.GNeuralFlow.experiments.latent_ode.lib.likelihood_eval import *
from layers.GNeuralFlow.experiments.latent_ode.lib.base_models import VAE_Baseline
from layers.GNeuralFlow.models.gnn import GNN
from layers.GNeuralFlow.models.mlp import MLP


class LatentODE(VAE_Baseline):
	def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver,
	             z0_prior, device, obsrv_std=None,
	             use_binary_classif=False, use_poisson_proc=False,
	             linear_classifier=False,
	             classif_per_tp=False,
	             n_labels=1,
	             train_classif_w_reconstr=False,
	             nfeats=1):

		super(LatentODE, self).__init__(
			input_dim=input_dim,
			latent_dim=latent_dim,
			z0_prior=z0_prior,
			device=device,
			obsrv_std=obsrv_std,
			use_binary_classif=use_binary_classif,
			classif_per_tp=classif_per_tp,
			linear_classifier=linear_classifier,
			use_poisson_proc=use_poisson_proc,
			n_labels=n_labels,
			train_classif_w_reconstr=train_classif_w_reconstr
		)

		self.encoder_z0 = encoder_z0
		self.diffeq_solver = diffeq_solver
		self.decoder = decoder
		self.use_poisson_proc = use_poisson_proc
		self.gcn = GNN(input_size=nfeats, hidden_size=32)
		self.nfeats = nfeats

	def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps,
		mask = None, n_traj_samples = 1, run_backwards = True, mode = None, adj = None):

		assert isinstance(self.encoder_z0, Encoder_z0_ODE_RNN)

		if mask is not None:
			first_point_mu, first_point_std = self.encoder_z0(truth, mask, truth_time_steps, adj, run_backwards)

			means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
			sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
			first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

		assert(torch.sum(first_point_std < 0) == 0.)

		if self.use_poisson_proc:
			n_traj_samples, n_traj, n_dims = first_point_enc.size()
			zeros = torch.zeros([n_traj_samples, n_traj,self.input_dim]).to(truth)
			first_point_enc_aug = torch.cat((first_point_enc, zeros), -1)
		else:
			first_point_enc_aug = first_point_enc

		assert(not torch.isnan(time_steps_to_predict).any())
		assert(not torch.isnan(first_point_enc).any())
		assert(not torch.isnan(first_point_enc_aug).any())

		# sol_y shape [n_traj_samples, n_samples, n_timepoints, n_latents]
		initial_state = first_point_enc_aug.unsqueeze(-2)
		sol_y = self.diffeq_solver(x=initial_state, h=None, t=time_steps_to_predict.unsqueeze(0))

		if self.use_poisson_proc:
			sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

			assert(torch.sum(int_lambda[:,:,0,:]) == 0.)
			assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

		pred_x = self.decoder(sol_y)

		all_extra_info = {
			'first_point': (first_point_mu, first_point_std, first_point_enc),
			'latent_traj': sol_y.detach()
		}

		if self.use_poisson_proc:
			all_extra_info['int_lambda'] = int_lambda[:,:,-1,:]
			all_extra_info['log_lambda_y'] = log_lambda_y

		if self.use_binary_classif:
			if self.classif_per_tp:
				all_extra_info['label_predictions'] = self.classifier(sol_y)
			else:
				all_extra_info['label_predictions'] = self.classifier(first_point_enc).squeeze(-1)

		return pred_x, all_extra_info


	def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples = 1):
		starting_point_enc = self.z0_prior.sample([n_traj_samples, 1, self.latent_dim]).squeeze(-1)

		starting_point_enc_aug = starting_point_enc
		if self.use_poisson_proc:
			n_traj_samples, n_traj, _ = starting_point_enc.size()
			zeros = torch.zeros(n_traj_samples, n_traj,self.input_dim).to(time_steps_to_predict)
			starting_point_enc_aug = torch.cat((starting_point_enc, zeros), -1)

		sol_y = self.diffeq_solver.sample_traj_from_prior(starting_point_enc_aug, time_steps_to_predict,
			n_traj_samples = 3)

		if self.use_poisson_proc:
			sol_y, _, _, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

		return self.decoder(sol_y)
