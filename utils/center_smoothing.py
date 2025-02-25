# Adapted from the original center smoothing paper:
# Center Smoothing: Provable Robustness for Functions with Metric-Space Outputs
# Accompanying source code: https://github.com/aounon/center-smoothing/blob/main/center_smoothing.py

import math

import numpy as np
# import torch
import tensorlayerx as tlx
from scipy.stats import norm

from .sampler import GaussianNoiseAdder
import paddle
import torch
import tensorflow as tf


def repeat_along_dim(t, num_repeat: int, dim: int = 0):
    return tlx.ops.tile(tlx.expand_dims(t, axis=dim), [1] * dim + [num_repeat] + [1] * (t.ndim - dim - 1))


def l2_dist(batch1, batch2):
    dist = tlx.ops.sqrt(tlx.ops.reduce_sum(tlx.ops.square(batch1 - batch2), axis=1))
    return dist.numpy()


class CenterSmoothing:
    ABSTAIN = -1.0

    def __init__(self, attribute_vector, fair_adapter,
                 sigma: float, dist_fn=l2_dist, n_pred: int = 10 ** 4, n_cert: int = 10 ** 6,
                 n_cntr: int = 30, alpha_1: float = 0.005, alpha_2: float = 0.005,
                 triang_delta: float = 0.05, radius_coeff: int = 3, output_is_hd: bool = False):
        self.fair_adapter = fair_adapter
        self.noise_adder = GaussianNoiseAdder(attribute_vector, sigma)

        self.dist_fn = dist_fn
        self.sigma = sigma
        self.n_pred = n_pred  # Number of samples used for prediction (center computation)
        self.n_cert = n_cert  # Number of samples used for certification
        self.n_cntr = n_cntr  # Number of candidate centers
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.triang_delta = triang_delta
        self.radius_coeff = radius_coeff  # for the relaxed triangle inequality
        self.output_is_hd = output_is_hd  # whether to use procedure for high-dimensional outputs


    def compute_center(self, single_x, batch_size: int = 1000):
        if tlx.BACKEND == 'tensorflow':
            context_manager = tf.GradientTape(persistent=True)
        elif tlx.BACKEND == 'torch':
            context_manager = torch.no_grad()
        elif tlx.BACKEND == 'paddle':
            context_manager = paddle.no_grad()
        else:
            raise NotImplementedError("Unsupported backend")
        
        with context_manager:
            delta_1 = math.sqrt(math.log(2 / self.alpha_1) / (2 * self.n_pred))
            is_good = False
            num = self.n_pred

            single_h_latent = tlx.ops.stop_gradient(single_x)
            batch_h_latent = repeat_along_dim(single_h_latent, batch_size)
            samples = None

            while num > 0:
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                if this_batch_size != batch_size:
                    batch_h_latent = repeat_along_dim(single_h_latent, this_batch_size)

                z_batch = self.base_function(batch_h_latent, add_noise=True)

                if samples is None:
                    samples = z_batch
                else:
                    samples = tlx.concat((samples, z_batch), axis=0)

            center, radius = self.meb(samples)
            num_pts = self.pts_in_nbd(single_h_latent, center, radius, batch_size=batch_size)

            frac = num_pts / self.n_pred
            p_delta_1 = frac - delta_1
            delta_2 = (1 / 2) - p_delta_1


            if max(delta_1, delta_2) <= self.triang_delta:
                is_good = True
            else:
                # print('Bad center. Abstaining ...')
                pass

        return center, is_good

    def compute_center_hd(self, input, batch_size: int = 1000):
        raise NotImplementedError('`CenterSmoothing.compute_center_hd` is not currently used')

    def pts_in_nbd(self, single_h_latent, center, radius: float,
                   batch_size: int = 1000):
        if tlx.BACKEND == 'tensorflow':
            context_manager = tf.GradientTape(persistent=True)
        elif tlx.BACKEND == 'torch':
            context_manager = torch.no_grad()
        elif tlx.BACKEND == 'paddle':
            context_manager = paddle.no_grad()
        else:
            raise NotImplementedError("Unsupported backend")

        with context_manager:
            while num > 0:
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                if this_batch_size != batch_size:
                    batch_h_latent = repeat_along_dim(single_h_latent, this_batch_size)
                    batch_cen = repeat_along_dim(center, this_batch_size)

                samples = self.base_function(batch_h_latent, add_noise=True)

                dist = tlx.ops.sqrt(tlx.ops.reduce_sum(tlx.ops.square(samples - batch_cen), axis=1))
                num_pts += int(tlx.ops.reduce_sum(tlx.ops.where(dist <= radius, 1, 0)))
        return num_pts

    def meb(self, samples):
        if tlx.BACKEND == 'tensorflow':
            context_manager = tf.GradientTape(persistent=True)
        elif tlx.BACKEND == 'torch':
            context_manager = torch.no_grad()
        elif tlx.BACKEND == 'paddle':
            context_manager = paddle.no_grad()
        else:
            raise NotImplementedError("Unsupported backend")

        with context_manager:
            dist = tlx.ops.sqrt(tlx.ops.reduce_sum(tlx.ops.square(samples.expand_dims(1) - samples), axis=2))
            median_dist = np.quantile(dist.numpy(), q=0.5, axis=1)
            # min_value, min_index = tlx.ops.min(median_dist, axis=0)
            min_value, min_index = np.min(median_dist), np.argmin(median_dist)
            radius = min_value
            center = samples[min_index]
        return center, radius

    def base_function(self, batch_h_latent, add_noise: bool = True):
        if add_noise:
            z_noise_latent_batch = self.noise_adder.add_noise(batch_h_latent)
        else:
            z_noise_latent_batch = batch_h_latent

        if tlx.BACKEND == 'tensorflow':
            stop_gradient = tf.stop_gradient
        elif tlx.BACKEND == 'torch':
            stop_gradient = lambda x: x.detach()
        elif tlx.BACKEND == 'paddle':
            stop_gradient = paddle.stop_gradient
        else:
            raise NotImplementedError("Unsupported backend")

        adapter_input = stop_gradient(z_noise_latent_batch)
        return self.fair_adapter(adapter_input)
