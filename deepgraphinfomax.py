import copy
from typing import Callable, Any

import numpy as np
import tensorlayerx as tlx
from tensorlayerx import nn
from tensorlayerx.nn import Module
import tensorflow as tf

EPS = 1e-15

class DeepGraphInfomax(Module):
    def __init__(
        self,
        hidden_channels: int,
        encoder: Module,
        summary: Callable,
        corruption: Callable,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption

        self.weight = self._get_weights('weight', shape=(hidden_channels, hidden_channels), init=tlx.initializers.random_uniform())
        # self.weight = tlx.nn.Parameter(tlx.initializers.random_uniform((self.hidden_channels, self.hidden_channels)))
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        # Reset encoder and summary components (initialize their parameters)
        self._reset(self.encoder)
        self._reset(self.summary)
        
        # Initialize the weight parameter using uniform distribution
        # self.weight.assign(tlx.initializers.random_uniform((self.hidden_channels, self.hidden_channels)))
        self.weight = self._get_weights('weight', shape=(self.hidden_channels, self.hidden_channels), init=tlx.initializers.random_uniform())


    def _reset(self, value: Any):
        """Helper function to reset parameters recursively."""
        if hasattr(value, 'reset_parameters'):
            value.reset_parameters()
        else:
            for child in value.children() if hasattr(value, 'children') else []:
                self._reset(child)

    # def reset_parameters(self):
    #     # # if hasattr(self.encoder, 'reset_parameters'):
    #     # #     self.encoder.reset_parameters()
    #     # # else:
    #     # #     for name, param in self.encoder.named_parameters():
    #     # #         if "weight" in name:
    #     # #             tlx.initializers.random_uniform()(param)
    #     # #         elif "bias" in name and param is not None:
    #     # #             tlx.initializers.zeros()(param)
    #     # if hasattr(self.encoder, 'reset_parameters'):
    #     #     self.encoder.reset_parameters()
    #     # else:
    #     #     # 使用 trainable_weights 初始化参数
    #     #     for param in self.encoder.trainable_weights:
    #     #         if param.name.endswith("weight"):
    #     #             shape = tuple(map(int, param.shape))
    #     #             param.assign(tlx.initializers.random_uniform()(shape=shape))
    #     #         elif param.name.endswith("bias") and param is not None:
    #     #             # tlx.initializers.zeros()(param)
    #     #             param.assign(tlx.initializers.zeros()(shape=param.shape))

    #     # if callable(self.summary) and hasattr(self.summary, 'reset_parameters'):
    #     #     self.summary.reset_parameters()

    #     # if hasattr(self, "weight"):
    #     #     tlx.initializers.random_uniform()(self.weight)
    #     if hasattr(self.encoder, 'reset_parameters'):
    #         self.encoder.reset_parameters()
    #     if hasattr(self.summary, 'reset_parameters'):
    #         self.summary.reset_parameters()

    #     # Initialize the weight using uniform distribution
    #     self.weight.assign(tlx.random.uniform((self.hidden_channels, self.hidden_channels), dtype=tlx.float32))

    def forward(self, *args, **kwargs):
        pos_z = self.encoder(*args, **kwargs)

        cor = self.corruption(*args, **kwargs)
        cor = cor if isinstance(cor, tuple) else (cor, )
        cor_args = cor[:len(args)]
        cor_kwargs = copy.copy(kwargs)
        for key, value in zip(kwargs.keys(), cor[len(args):]):
            cor_kwargs[key] = value

        neg_z = self.encoder(*cor_args, **cor_kwargs)

        summary = self.summary(pos_z, *args, **kwargs)

        return pos_z, neg_z, summary

    def discriminate(self, z, summary,
                     sigmoid: bool = True):
        summary = tlx.ops.transpose(summary) if summary.ndim > 1 else summary
        value = tlx.ops.matmul(z, tlx.ops.matmul(self.weight, summary))
        return tlx.ops.sigmoid(value) if sigmoid else value

    def loss(self, pos_z, neg_z, summary):
        r"""Computes the mutual information maximization objective."""
        pos_loss = -tlx.ops.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -tlx.ops.log(1 -
                                self.discriminate(neg_z, summary, sigmoid=True) +
                                EPS).mean()

        return pos_loss + neg_loss

    def test(
        self,
        train_z,
        train_y,
        test_z,
        test_y,
        solver: str = 'lbfgs',
        *args,
        **kwargs,
    ) -> float:
        r"""Evaluates latent space quality via a logistic regression downstream
        task.
        """
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(solver=solver, *args,
                                 **kwargs).fit(train_z.numpy(),
                                               train_y.numpy())
        return clf.score(test_z.numpy(),
                         test_y.numpy())

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.hidden_channels})'