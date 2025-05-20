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
# This source code is derived from Pytorch RKN Implementation (https://github.com/ALRhub/rkn_share)
# Copyright (c) 2021 Philipp Becker (Autonomous Learning Robots Lab @ KIT)
# licensed under MIT License
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import numpy as np
from PIL import Image
from PIL import ImageDraw
import os


# taken from https://github.com/ALRhub/rkn_share/ and not modified
def add_img_noise(imgs, first_n_clean, random, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75, t_uu=1.0):
    """
    :param imgs: Images to add noise to
    :param first_n_clean: Keep first_n_images clean to allow the filter to burn in
    :param random: np.random.RandomState used for sampling
    :param r: "correlation (over time) factor" the smaller the more the noise is correlated
    :param t_ll: lower bound of the interval the lower bound for each sequence is sampled from
    :param t_lu: upper bound of the interval the lower bound for each sequence is sampled from
    :param t_ul: lower bound of the interval the upper bound for each sequence is sampled from
    :param t_uu: upper bound of the interval the upper bound for each sequence is sampled from
    :return: noisy images, factors used to create them
    """

    assert t_ll <= t_lu <= t_ul <= t_uu, "Invalid bounds for noise generation"
    if len(imgs.shape) < 5:
        imgs = np.expand_dims(imgs, -1)
    batch_size, seq_len = imgs.shape[:2]
    factors = np.zeros([batch_size, seq_len])
    factors[:, 0] = random.uniform(low=0.0, high=1.0, size=batch_size)
    for i in range(seq_len - 1):
        factors[:, i + 1] = np.clip(factors[:, i] + random.uniform(
            low=-r, high=r, size=batch_size), a_min=0.0, a_max=1.0)

    t1 = random.uniform(low=t_ll, high=t_lu, size=(batch_size, 1))
    t2 = random.uniform(low=t_ul, high=t_uu, size=(batch_size, 1))

    factors = (factors - t1) / (t2 - t1)
    factors = np.clip(factors, a_min=0.0, a_max=1.0)
    factors = np.reshape(factors, list(factors.shape) + [1, 1, 1])
    factors[:, :first_n_clean] = 1.0
    noisy_imgs = []

    for i in range(batch_size):
        if imgs.dtype == np.uint8:
            noise = random.uniform(low=0.0, high=255, size=imgs.shape[1:])
            noisy_imgs.append(
                (factors[i] * imgs[i] + (1 - factors[i]) * noise).astype(np.uint8))
        else:
            noise = random.uniform(low=0.0, high=1.1, size=imgs.shape[1:])
            noisy_imgs.append(factors[i] * imgs[i] + (1 - factors[i]) * noise)

    return np.squeeze(np.concatenate([np.expand_dims(n, 0) for n in noisy_imgs], 0)), factors


# taken from https://github.com/ALRhub/rkn_share/ and not modified
def add_img_noise4(imgs, first_n_clean, random, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75, t_uu=1.0):
    """
    :param imgs: Images to add noise to
    :param first_n_clean: Keep first_n_images clean to allow the filter to burn in
    :param random: np.random.RandomState used for sampling
    :param r: "correlation (over time) factor" the smaller the more the noise is correlated
    :param t_ll: lower bound of the interval the lower bound for each sequence is sampled from
    :param t_lu: upper bound of the interval the lower bound for each sequence is sampled from
    :param t_ul: lower bound of the interval the upper bound for each sequence is sampled from
    :param t_uu: upper bound of the interval the upper bound for each sequence is sampled from
    :return: noisy images, factors used to create them
    """

    half_x = int(imgs.shape[2] / 2)
    half_y = int(imgs.shape[3] / 2)
    assert t_ll <= t_lu <= t_ul <= t_uu, "Invalid bounds for noise generation"
    if len(imgs.shape) < 5:
        imgs = np.expand_dims(imgs, -1)
    batch_size, seq_len = imgs.shape[:2]
    factors = np.zeros([batch_size, seq_len, 4])
    factors[:, 0] = random.uniform(low=0.0, high=1.0, size=(batch_size, 4))
    for i in range(seq_len - 1):
        factors[:, i + 1] = np.clip(factors[:, i] + random.uniform(
            low=-r, high=r, size=(batch_size, 4)), a_min=0.0, a_max=1.0)

    t1 = random.uniform(low=t_ll, high=t_lu, size=(batch_size, 1, 4))
    t2 = random.uniform(low=t_ul, high=t_uu, size=(batch_size, 1, 4))

    factors = (factors - t1) / (t2 - t1)
    factors = np.clip(factors, a_min=0.0, a_max=1.0)
    factors = np.reshape(factors, list(factors.shape) + [1, 1, 1])
    factors[:, :first_n_clean] = 1.0
    noisy_imgs = []
    qs = []
    for i in range(batch_size):
        if imgs.dtype == np.uint8:
            qs.append(detect_pendulums(imgs[i], half_x, half_y))
            noise = random.uniform(low=0.0, high=255, size=[
                                   4, seq_len, half_x, half_y, imgs.shape[-1]]).astype(np.uint8)
            curr = np.zeros(imgs.shape[1:], dtype=np.uint8)
            curr[:, :half_x, :half_y] = (factors[i, :, 0] * imgs[i, :, :half_x, :half_y] + (
                1 - factors[i, :, 0]) * noise[0]).astype(np.uint8)
            curr[:, :half_x, half_y:] = (factors[i, :, 1] * imgs[i, :, :half_x, half_y:] + (
                1 - factors[i, :, 1]) * noise[1]).astype(np.uint8)
            curr[:, half_x:, :half_y] = (factors[i, :, 2] * imgs[i, :, half_x:, :half_y] + (
                1 - factors[i, :, 2]) * noise[2]).astype(np.uint8)
            curr[:, half_x:, half_y:] = (factors[i, :, 3] * imgs[i, :, half_x:, half_y:] + (
                1 - factors[i, :, 3]) * noise[3]).astype(np.uint8)
        else:
            noise = random.uniform(low=0.0, high=1.0, size=[
                                   4, seq_len, half_x, half_y, imgs.shape[-1]])
            curr = np.zeros(imgs.shape[1:])
            curr[:, :half_x, :half_y] = factors[i, :, 0] * imgs[i, :,
                                                                :half_x, :half_y] + (1 - factors[i, :, 0]) * noise[0]
            curr[:, :half_x, half_y:] = factors[i, :, 1] * imgs[i, :,
                                                                :half_x, half_y:] + (1 - factors[i, :, 1]) * noise[1]
            curr[:, half_x:, :half_y] = factors[i, :, 2] * imgs[i, :,
                                                                half_x:, :half_y] + (1 - factors[i, :, 2]) * noise[2]
            curr[:, half_x:, half_y:] = factors[i, :, 3] * imgs[i, :,
                                                                half_x:, half_y:] + (1 - factors[i, :, 3]) * noise[3]
        noisy_imgs.append(curr)

    factors_ext = np.concatenate([np.squeeze(factors), np.zeros(
        [factors.shape[0], factors.shape[1], 1])], -1)
    q = np.concatenate([np.expand_dims(q, 0) for q in qs], 0)
    f = np.zeros(q.shape)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            for k in range(3):
                f[i, j, k] = factors_ext[i, j, q[i, j, k]]

    return np.squeeze(np.concatenate([np.expand_dims(n, 0) for n in noisy_imgs], 0)), f


# taken from https://github.com/ALRhub/rkn_share/ and not modified
def detect_pendulums(imgs, half_x, half_y):
    qs = [imgs[:, :half_x, :half_y], imgs[:, :half_x, half_y:],
          imgs[:, half_x:, :half_y], imgs[:, half_x:, half_y:]]

    r_cts = np.array(
        [np.count_nonzero(q[:, :, :, 0] > 5, axis=(-1, -2)) for q in qs]).T
    g_cts = np.array(
        [np.count_nonzero(q[:, :, :, 1] > 5, axis=(-1, -2)) for q in qs]).T
    b_cts = np.array(
        [np.count_nonzero(q[:, :, :, 2] > 5, axis=(-1, -2)) for q in qs]).T

    cts = np.concatenate([np.expand_dims(c, 1)
                         for c in [r_cts, g_cts, b_cts]], 1)

    q_max = np.max(cts, -1)
    q = np.argmax(cts, -1)
    q[q_max < 10] = 4
    return q


# taken from https://github.com/ALRhub/rkn_share/ and not modified
class Pendulum:

    MAX_VELO_KEY = 'max_velo'
    MAX_TORQUE_KEY = 'max_torque'
    MASS_KEY = 'mass'
    LENGTH_KEY = 'length'
    GRAVITY_KEY = 'g'
    FRICTION_KEY = 'friction'
    DT_KEY = 'dt'
    SIM_DT_KEY = 'sim_dt'
    TRANSITION_NOISE_TRAIN_KEY = 'transition_noise_train'
    TRANSITION_NOISE_TEST_KEY = 'transition_noise_test'

    OBSERVATION_MODE_LINE = "line"
    OBSERVATION_MODE_BALL = "ball"


    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def __init__(self,
                 img_size,
                 observation_mode,
                 generate_actions=False,
                 transition_noise_std=0.0,
                 observation_noise_std=0.0,
                 pendulum_params=None,
                 seed=0):

        assert observation_mode == Pendulum.OBSERVATION_MODE_BALL or observation_mode == Pendulum.OBSERVATION_MODE_LINE
        # Global Parameters
        self.state_dim = 2
        self.action_dim = 1
        self.img_size = img_size
        self.observation_dim = img_size ** 2
        self.observation_mode = observation_mode

        self.random = np.random.RandomState(seed)

        # image parameters
        self.img_size_internal = 128
        self.x0 = self.y0 = 64
        self.plt_length = 55 if self.observation_mode == Pendulum.OBSERVATION_MODE_LINE else 50
        self.plt_width = 8

        self.generate_actions = generate_actions

        # simulation parameters
        if pendulum_params is None:
            pendulum_params = self.pendulum_default_params()
        self.max_velo = pendulum_params[Pendulum.MAX_VELO_KEY]
        self.max_torque = pendulum_params[Pendulum.MAX_TORQUE_KEY]
        self.dt = pendulum_params[Pendulum.DT_KEY]
        self.mass = pendulum_params[Pendulum.MASS_KEY]
        self.length = pendulum_params[Pendulum.LENGTH_KEY]
        self.inertia = self.mass * self.length**2 / 3
        self.g = pendulum_params[Pendulum.GRAVITY_KEY]
        self.friction = pendulum_params[Pendulum.FRICTION_KEY]
        self.sim_dt = pendulum_params[Pendulum.SIM_DT_KEY]

        self.observation_noise_std = observation_noise_std
        self.transition_noise_std = transition_noise_std

        self.tranisition_covar_mat = np.diag(
            np.array([1e-8, self.transition_noise_std**2, 1e-8, 1e-8]))
        self.observation_covar_mat = np.diag(
            [self.observation_noise_std**2, self.observation_noise_std**2])

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def sample_data_set(self, num_episodes, episode_length, full_targets):
        states = np.zeros((num_episodes, episode_length, self.state_dim))
        actions = self._sample_action(
            (num_episodes, episode_length, self.action_dim))
        states[:, 0, :] = self._sample_init_state(num_episodes)
        t = np.zeros((num_episodes, episode_length))

        for i in range(1, episode_length):
            states[:, i, :], dt = self._get_next_states(
                states[:, i - 1, :], actions[:, i - 1, :])
            t[:, i:] += dt
        states[..., 0] -= np.pi

        if self.observation_noise_std > 0.0:
            observation_noise = self.random.normal(loc=0.0,
                                                   scale=self.observation_noise_std,
                                                   size=states.shape)
        else:
            observation_noise = np.zeros(states.shape)

        targets = self.pendulum_kinematic(states)
        if self.observation_mode == Pendulum.OBSERVATION_MODE_LINE:
            noisy_states = states + observation_noise
            noisy_targets = self.pendulum_kinematic(noisy_states)
        elif self.observation_mode == Pendulum.OBSERVATION_MODE_BALL:
            noisy_targets = targets + observation_noise
        imgs = self._generate_images(noisy_targets[..., :2])

        return imgs, targets[..., :(4 if full_targets else 2)], states, noisy_targets[..., :(4 if full_targets else 2)], t/self.dt

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    @staticmethod
    def pendulum_default_params():
        return {
            Pendulum.MAX_VELO_KEY: 8,
            Pendulum.MAX_TORQUE_KEY: 10,
            Pendulum.MASS_KEY: 1,
            Pendulum.LENGTH_KEY: 1,
            Pendulum.GRAVITY_KEY: 9.81,
            Pendulum.FRICTION_KEY: 0,
            Pendulum.DT_KEY: 0.05,
            Pendulum.SIM_DT_KEY: 1e-4
        }

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _sample_action(self, shape):
        if self.generate_actions:
            return self.random.uniform(-self.max_torque, self.max_torque, shape)
        else:
            return np.zeros(shape=shape)

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _transition_function(self, states, actions):
        dt = self.dt
        n_steps = dt / self.sim_dt

        if n_steps != np.round(n_steps):
            #print(n_steps, 'Warning from Pendulum: dt does not match up')
            n_steps = np.round(n_steps)

        c = self.g * self.length * self.mass / self.inertia
        for i in range(0, int(n_steps)):
            velNew = states[..., 1:2] + self.sim_dt * (c * np.sin(states[..., 0:1])
                                                       + actions / self.inertia
                                                       - states[..., 1:2] * self.friction)
            states = np.concatenate(
                (states[..., 0:1] + self.sim_dt * velNew, velNew), axis=1)
        return states, dt

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _get_next_states(self, states, actions):
        actions = np.maximum(-self.max_torque,
                             np.minimum(actions, self.max_torque))

        states, dt = self._transition_function(states, actions)
        if self.transition_noise_std > 0.0:
            states[:, 1] += self.random.normal(loc=0.0,
                                               scale=self.transition_noise_std,
                                               size=[len(states)])

        states[:, 0] = ((states[:, 0]) % (2 * np.pi))
        return states, dt

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def get_ukf_smothing(self, obs):
        batch_size, seq_length = obs.shape[:2]
        succ = np.zeros(batch_size, dtype=np.bool)
        means = np.zeros([batch_size, seq_length, 4])
        covars = np.zeros([batch_size, seq_length, 4, 4])
        fail_ct = 0
        for i in range(batch_size):
            if i % 10 == 0:
                print(i)
            try:
                means[i], covars[i] = self.ukf.filter(obs[i])
                succ[i] = True
            except:
                fail_ct += 1
        print(fail_ct / batch_size, "failed")

        return means[succ], covars[succ], succ

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _sample_init_state(self, nr_epochs):
        return np.concatenate((self.random.uniform(0, 2 * np.pi, (nr_epochs, 1)), np.zeros((nr_epochs, 1))), 1)

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def add_observation_noise(self, imgs, first_n_clean, r=0.2, t_ll=0.1, t_lu=0.4, t_ul=0.6, t_uu=0.9):
        return add_img_noise(imgs, first_n_clean, self.random, r, t_ll, t_lu, t_ul, t_uu)

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _get_task_space_pos(self, joint_states):
        task_space_pos = np.zeros(list(joint_states.shape[:-1]) + [2])
        task_space_pos[..., 0] = np.sin(joint_states[..., 0]) * self.length
        task_space_pos[..., 1] = np.cos(joint_states[..., 0]) * self.length
        return task_space_pos

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _generate_images(self, ts_pos):
        imgs = np.zeros(shape=list(ts_pos.shape)[
                        :-1] + [self.img_size, self.img_size], dtype=np.uint8)
        for seq_idx in range(ts_pos.shape[0]):
            for idx in range(ts_pos.shape[1]):
                imgs[seq_idx, idx] = self._generate_single_image(
                    ts_pos[seq_idx, idx])

        return imgs

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _generate_single_image(self, pos):
        x1 = pos[0] * (self.plt_length / self.length) + self.x0
        y1 = pos[1] * (self.plt_length / self.length) + self.y0
        img = Image.new('F', (self.img_size_internal,
                        self.img_size_internal), 0.0)
        draw = ImageDraw.Draw(img)
        if self.observation_mode == Pendulum.OBSERVATION_MODE_LINE:
            draw.line([(self.x0, self.y0), (x1, y1)],
                      fill=1.0, width=self.plt_width)
        elif self.observation_mode == Pendulum.OBSERVATION_MODE_BALL:
            x_l = x1 - self.plt_width
            x_u = x1 + self.plt_width
            y_l = y1 - self.plt_width
            y_u = y1 + self.plt_width
            draw.ellipse((x_l, y_l, x_u, y_u), fill=1.0)

        img = img.resize((self.img_size, self.img_size),
                         resample=Image.ANTIALIAS)
        img_as_array = np.asarray(img)
        img_as_array = np.clip(img_as_array, 0, 1)
        return 255.0 * img_as_array

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _kf_transition_function(self, state, noise):
        nSteps = self.dt / self.sim_dt

        if nSteps != np.round(nSteps):
            print('Warning from Pendulum: dt does not match up')
            nSteps = np.round(nSteps)

        c = self.g * self.length * self.mass / self.inertia
        for i in range(0, int(nSteps)):
            velNew = state[1] + self.sim_dt * \
                (c * np.sin(state[0]) - state[1] * self.friction)
            state = np.array([state[0] + self.sim_dt * velNew, velNew])
        state[0] = state[0] % (2 * np.pi)
        state[1] = state[1] + noise[1]
        return state

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def pendulum_kinematic_single(self, js):
        theta, theat_dot = js
        x = np.sin(theta)
        y = np.cos(theta)
        x_dot = theat_dot * y
        y_dot = theat_dot * -x
        return np.array([x, y, x_dot, y_dot]) * self.length

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def pendulum_kinematic(self, js_batch):
        theta = js_batch[..., :1]
        theta_dot = js_batch[..., 1:]
        x = np.sin(theta)
        y = np.cos(theta)
        x_dot = theta_dot * y
        y_dot = theta_dot * -x
        return np.concatenate([x, y, x_dot, y_dot], axis=-1)

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def inverse_pendulum_kinematics(self, ts_batch):
        x = ts_batch[..., :1]
        y = ts_batch[..., 1:2]
        x_dot = ts_batch[..., 2:3]
        y_dot = ts_batch[..., 3:]
        val = x / y
        theta = np.arctan2(x, y)
        theta_dot_outer = 1 / (1 + val**2)
        theta_dot_inner = (x_dot * y - y_dot * x) / y**2
        return np.concatenate([theta, theta_dot_outer * theta_dot_inner], axis=-1)


# taken from https://github.com/ALRhub/rkn_share/ and modified
def generate_pendulums(file_path, task, impute_rate=0.5):
    
    if task == 'interpolation':
        pend_params = Pendulum.pendulum_default_params()
        pend_params[Pendulum.FRICTION_KEY] = 0.1
        n = 100

        pendulum = Pendulum(24, observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                            transition_noise_std=0.1, observation_noise_std=1e-5,
                            seed=42, pendulum_params=pend_params)
        rng = pendulum.random

        train_obs, _, _, _, train_ts = pendulum.sample_data_set(
            2000, n, full_targets=False)
        train_obs = np.expand_dims(train_obs, -1)
        train_targets = train_obs.copy()
        train_obs_valid = rng.rand(
            train_obs.shape[0], train_obs.shape[1], 1) > impute_rate
        train_obs_valid[:, :5] = True
        train_obs[np.logical_not(np.squeeze(train_obs_valid))] = 0

        test_obs, _, _, _, test_ts = pendulum.sample_data_set(
            1000, n, full_targets=False)
        test_obs = np.expand_dims(test_obs, -1)
        test_targets = test_obs.copy()
        test_obs_valid = rng.rand(
            test_obs.shape[0], test_obs.shape[1], 1) > impute_rate
        test_obs_valid[:, :5] = True
        test_obs[np.logical_not(np.squeeze(test_obs_valid))] = 0

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        np.savez_compressed(os.path.join(file_path, f"pend_interpolation_ir{impute_rate}"),
                            train_obs=train_obs, train_targets=train_targets, train_obs_valid=train_obs_valid, train_ts=train_ts,
                            test_obs=test_obs, test_targets=test_targets, test_obs_valid=test_obs_valid, test_ts=test_ts)
    
    elif task == 'regression':
        pend_params = Pendulum.pendulum_default_params()
        pend_params[Pendulum.FRICTION_KEY] = 0.1
        pend_params[Pendulum.DT_KEY] = 0.01
        n = 100

        pendulum = Pendulum(24, observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                            transition_noise_std=0.1, observation_noise_std=1e-5,
                            seed=42, pendulum_params=pend_params)

        train_obs, train_targets, _, _, train_ts = pendulum.sample_data_set(
            2000, n, full_targets=False)
        train_obs, _ = pendulum.add_observation_noise(train_obs, first_n_clean=5, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75,
                                                      t_uu=1.0)
        train_obs = np.expand_dims(train_obs, -1)

        test_obs, test_targets, _, _, test_ts = pendulum.sample_data_set(
            1000, n, full_targets=False)
        test_obs, _ = pendulum.add_observation_noise(test_obs, first_n_clean=5, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75,
                                                     t_uu=1.0)
        test_obs = np.expand_dims(test_obs, -1)

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        np.savez_compressed(os.path.join(file_path, f"pend_regression.npz"),
                            train_obs=train_obs, train_targets=train_targets, train_ts=train_ts,
                            test_obs=test_obs, test_targets=test_targets, test_ts=test_ts)
