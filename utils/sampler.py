from abc import ABC, abstractmethod

import tensorlayerx as tlx
import paddle
import torch
import tensorflow as tf


class NoiseAdder(ABC):

    def __init__(self, attribute_vector):
        self.attribute_vector = attribute_vector
        self.cur_batch_size = -1
        self.attribute_vector_repeated = None

    def set_attribute_vectors_repeated(self, required_batch_size: int):
        if self.cur_batch_size != required_batch_size:
            self.cur_batch_size = required_batch_size
            self.attribute_vector_repeated = tlx.ops.tile(self.attribute_vector.expand_dims(0),
                                                          [self.cur_batch_size, 1])
    def add_noise(self, z_encoder):
        self.set_attribute_vectors_repeated(z_encoder.shape[0])
        return self._add_noise(z_encoder)

    @abstractmethod
    def _add_noise(self, z_gen_model_latents):
        pass


class GaussianNoiseAdder(NoiseAdder):

    def __init__(self, attribute_vector, sigma: float):
        super(GaussianNoiseAdder, self).__init__(attribute_vector)
        self.sigma = sigma

    def _add_noise(self, z_gen_model_latents):
        if tlx.BACKEND == 'tensorflow':
            stop_gradient = tf.stop_gradient
        elif tlx.BACKEND == 'torch':
            stop_gradient = lambda x: x.detach()
        elif tlx.BACKEND == 'paddle':
            stop_gradient = paddle.stop_gradient
        noisy_latents = stop_gradient(z_gen_model_latents)
        coeffs = tlx.ops.random_normal([self.cur_batch_size, 1], mean=0.0, stddev=self.sigma, dtype=noisy_latents.dtype)
        noisy_latents += self.attribute_vector_repeated * coeffs
        return noisy_latents
