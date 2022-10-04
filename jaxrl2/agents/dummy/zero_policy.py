"""Implementations of algorithms for continuous control."""
import matplotlib
matplotlib.use('Agg')

import numpy as np
from typing import Dict, Optional, Sequence, Tuple, Union
from jaxrl2.data.dataset import DatasetDict

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.agents.agent import Agent

class ZeroPolicy(Agent):

    def __init__(self,
                 seed: int,
                 observations: Union[jnp.ndarray, DatasetDict],
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 dropout_rate: Optional[float] = None,
                 encoder_type='resnet_34_v1',
                 encoder_norm='batch',
                 use_spatial_softmax=True,
                 softmax_temperature=1.0,
                 ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        self.action_dim = actions.shape[-1]

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        raise NotImplementedError

    @property
    def _save_dict(self):
        return {}

    def perform_eval(self, variant, i, wandb_logger, eval_buffer, eval_buffer_iterator, eval_env=None, do_control_eval=False):
        pass

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        action = np.zeros(self.action_dim)
        action[-1] = 1
        return action

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        action = np.zeros(self.action_dim)
        action[-1] = 1
        return action
