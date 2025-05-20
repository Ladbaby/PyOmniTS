# Modeling Irregular Time Series with Continuous Recurrent Units (CRUs)
# Copyright (c) 2022 Robert Bosch GmbH
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This source code is derived from Latent ODEs for Irregularly-Sampled Time Series (https://github.com/YuliaRubanova/latent_ode)
# Copyright (c) 2019 Yulia Rubanova
# licensed under MIT License
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import torch
from sklearn.model_selection import train_test_split
import os
import numpy as np
import tarfile
from torchvision.datasets.utils import download_url


# taken from https://github.com/YuliaRubanova/latent_ode and not modified
class PhysioNet(object):

	urls = [
		'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
		'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
	]

	outcome_urls = ['https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt']

	params = [
		'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
		'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
		'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
		'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
	]

	params_dict = {k: i for i, k in enumerate(params)}

	labels = [ "SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death" ]
	labels_dict = {k: i for i, k in enumerate(labels)}

	def __init__(self, root, train=True, download=False,
		quantization = 0.1, n_samples = None, device = torch.device("cpu")):

		self.root = root
		self.train = train
		self.reduce = "average"
		self.quantization = quantization

		if download:
			self.download()

		if not self._check_exists():
			raise RuntimeError('Dataset not found. You can use download=True to download it')

		if self.train:
			data_file = self.training_file
		else:
			data_file = self.test_file
		
		if device == torch.device("cpu"):
			self.data = torch.load(os.path.join(self.processed_folder, data_file), map_location='cpu')
			self.labels = torch.load(os.path.join(self.processed_folder, self.label_file), map_location='cpu')
		else:
			self.data = torch.load(os.path.join(self.processed_folder, data_file))
			self.labels = torch.load(os.path.join(self.processed_folder, self.label_file))

		if n_samples is not None:
			self.data = self.data[:n_samples]
			self.labels = self.labels[:n_samples]


	def download(self):
		if self._check_exists():
			return

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		os.makedirs(self.raw_folder, exist_ok=True)
		os.makedirs(self.processed_folder, exist_ok=True)

		# Download outcome data
		for url in self.outcome_urls:
			filename = url.rpartition('/')[2]
			download_url(url, self.raw_folder, filename, None)

			txtfile = os.path.join(self.raw_folder, filename)
			with open(txtfile) as f:
				lines = f.readlines()
				outcomes = {}
				for l in lines[1:]:
					l = l.rstrip().split(',')
					record_id, labels = l[0], np.array(l[1:]).astype(float)
					outcomes[record_id] = torch.Tensor(labels).to(self.device)

				torch.save(
					labels,
					os.path.join(self.processed_folder, filename.split('.')[0] + '.pt')
				)

		for url in self.urls:
			filename = url.rpartition('/')[2]
			download_url(url, self.raw_folder, filename, None)
			tar = tarfile.open(os.path.join(self.raw_folder, filename), "r:gz")
			tar.extractall(self.raw_folder)
			tar.close()

			print('Processing {}...'.format(filename))

			dirname = os.path.join(self.raw_folder, filename.split('.')[0])
			patients = []
			total = 0
			for txtfile in os.listdir(dirname):
				record_id = txtfile.split('.')[0]
				with open(os.path.join(dirname, txtfile)) as f:
					lines = f.readlines()
					prev_time = 0
					tt = [0.]
					vals = [torch.zeros(len(self.params)).to(self.device)]
					mask = [torch.zeros(len(self.params)).to(self.device)]
					nobs = [torch.zeros(len(self.params))]
					for l in lines[1:]:
						total += 1
						time, param, val = l.split(',')
						# Time in hours
						time = float(time.split(':')[0]) + float(time.split(':')[1]) / 60.
						# round up the time stamps (up to 6 min by default)
						# used for speed -- we actually don't need to quantize it in Latent ODE
						time = round(time / self.quantization) * self.quantization

						if time != prev_time:
							tt.append(time)
							vals.append(torch.zeros(len(self.params)).to(self.device))
							mask.append(torch.zeros(len(self.params)).to(self.device))
							nobs.append(torch.zeros(len(self.params)).to(self.device))
							prev_time = time

						if param in self.params_dict:
							#vals[-1][self.params_dict[param]] = float(val)
							n_observations = nobs[-1][self.params_dict[param]]
							if self.reduce == 'average' and n_observations > 0:
								prev_val = vals[-1][self.params_dict[param]]
								new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
								vals[-1][self.params_dict[param]] = new_val
							else:
								vals[-1][self.params_dict[param]] = float(val)
							mask[-1][self.params_dict[param]] = 1
							nobs[-1][self.params_dict[param]] += 1
						else:
							assert param == 'RecordID', 'Read unexpected param {}'.format(param)
				tt = torch.tensor(tt).to(self.device)
				vals = torch.stack(vals)
				mask = torch.stack(mask)

				labels = None
				if record_id in outcomes:
					# Only training set has labels
					labels = outcomes[record_id]
					# Out of 5 label types provided for Physionet, take only the last one -- mortality
					labels = labels[4]

				patients.append((record_id, tt, vals, mask, labels))

			torch.save(
				patients,
				os.path.join(self.processed_folder, 
					filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
			)
				
		print('Done!')

	def _check_exists(self):
		for url in self.urls:
			filename = url.rpartition('/')[2]

			if not os.path.exists(
				os.path.join(self.processed_folder, 
					filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
			):
				return False
		return True

	@property
	def raw_folder(self):
		return os.path.join(self.root, 'raw')

	@property
	def processed_folder(self):
		return os.path.join(self.root, 'processed')

	@property
	def training_file(self):
		return 'set-a_{}.pt'.format(self.quantization)

	@property
	def test_file(self):
		return 'set-b_{}.pt'.format(self.quantization)

	@property
	def label_file(self):
		return 'Outcomes-a.pt'

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

	def get_label(self, record_id):
		return self.labels[record_id]

	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		fmt_str += '    Split: {}\n'.format('train' if self.train is True else 'test')
		fmt_str += '    Root Location: {}\n'.format(self.root)
		fmt_str += '    Quantization: {}\n'.format(self.quantization)
		fmt_str += '    Reduce: {}\n'.format(self.reduce)
		return fmt_str


# new code component
def train_test_valid_split(input_path):
    a = torch.load(os.path.join(input_path, 'set-a_0.1.pt'))
    b = torch.load(os.path.join(input_path, 'set-b_0.1.pt'))
    data = a + b

    train_valid, test = train_test_split(data, test_size=0.2, random_state=0)
    train, valid = train_test_split(
        train_valid, test_size=0.25, random_state=0)
    return train, train_valid, valid, test


# new code component
def remove_timeinvariant_features(input_path, name):
    data = torch.load(os.path.join(input_path, name+'.pt'), map_location='cpu')
    data_timevariant = []
    for sample in data:
        obs = sample[2]
        mask = sample[3]
        obs_timevariant = obs[:, 4:]
        mask_timevariant = mask[:, 4:]
        data_timevariant.append(
            (sample[0], sample[1], obs_timevariant, mask_timevariant, sample[4]))
    return data_timevariant


# new code component
def normalize_data_and_save(input_path, name):
    data = torch.load(os.path.join(input_path, name+'.pt'), map_location='cpu')
    min_value, max_value = get_min_max_physionet(data)

    data_normalized = []
    for sample in data:
        obs = sample[2]
        mask = sample[3]
        obs_normalized = normalize_obs(obs, mask, min_value, max_value)
        data_normalized.append(
            (sample[0], sample[1], obs_normalized, sample[3], sample[4]))
    return data_normalized


# new code component
def normalize_obs(obs, mask, min_value, max_value):
    assert obs.shape[-1] == min_value.shape[-1] == max_value.shape[-1], 'Dimension missmatch'
    max_value[max_value == 0] = 1
    obs_norm = (obs - min_value) / (max_value - min_value)
    obs_norm[mask == 0] = 0
    return obs_norm


# new code component
def get_min_max_physionet(data):
    obs = torch.cat([sample[2] for sample in data])
    min_value, _ = torch.min(obs, dim=0)
    max_value, _ = torch.max(obs, dim=0)
    return min_value, max_value


def download_and_process_physionet(file_path):
    # initialize Physionet instance to download dataset
    dataset = PhysioNet(file_path, train=False, download=True)
    processed_path = os.path.join(file_path, 'processed')
    
    sets = ['train', 'train_valid', 'test', 'valid']
    train, train_valid, valid, test = train_test_valid_split(processed_path)
    torch.save(train, os.path.join(processed_path, 'train.pt'))
    torch.save(train_valid, os.path.join(processed_path, 'train_valid.pt'))
    torch.save(valid, os.path.join(processed_path, 'valid.pt'))
    torch.save(test, os.path.join(processed_path, 'test.pt'))

    for set in sets:
        data_timevariant = remove_timeinvariant_features(processed_path, name=set)
        torch.save(data_timevariant, os.path.join(processed_path, f'f37_{set}.pt'))

        data_normalized = normalize_data_and_save(processed_path, name=f'f37_{set}')
        torch.save(data_normalized, os.path.join(file_path, f'norm_{set}.pt'))

