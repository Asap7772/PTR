from typing import Dict, Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init, xavier_init, kaiming_init

from functools import partial
from typing import Any, Callable, Sequence, Tuple
import distrax
import wandb

ModuleDef = Any

import jax

from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, jax.lax.Precision, Tuple[str, str],
                      Tuple[jax.lax.Precision, jax.lax.Precision]]

default_kernel_init = nn.initializers.lecun_normal()


class SpatialSoftmax(nn.Module):
    height: int
    width: int
    channel: int
    pos_x: jnp.ndarray
    pos_y: jnp.ndarray
    temperature: None
    log_heatmap: bool = False

    @nn.compact
    def __call__(self, feature):
        if self.temperature == -1:
            from jax.nn import initializers
            # print("Trainable temperature parameter")
            temperature = self.param('softmax_temperature', initializers.ones, (1), jnp.float32)
        else:
            temperature = 1.

        # print(temperature)
        assert len(feature.shape) == 4
        batch_size, num_featuremaps = feature.shape[0], feature.shape[3]
        feature = feature.transpose(0, 3, 1, 2).reshape(batch_size, num_featuremaps, self.height * self.width)

        softmax_attention = nn.softmax(feature / temperature)
        expected_x = jnp.sum(self.pos_x * softmax_attention, axis=2, keepdims=True).reshape(batch_size, num_featuremaps)
        expected_y = jnp.sum(self.pos_y * softmax_attention, axis=2, keepdims=True).reshape(batch_size, num_featuremaps)
        expected_xy = jnp.concatenate([expected_x, expected_y], axis=1)

        expected_xy = jnp.reshape(expected_xy, [batch_size, 2*num_featuremaps])
        return expected_xy


class SpatialLearnedEmbeddings(nn.Module):
    height: int
    width: int
    channel: int
    num_features: int = 5
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    param_dtype: Dtype = jnp.float32
    
    @nn.compact
    def __call__(self, features):
        """ 
        features is B x H x W X C
        """
        kernel = self.param('kernel',
                            self.kernel_init,
                            (self.height, self.width, self.channel, self.num_features),
                            self.param_dtype)
        
        batch_size = features.shape[0]
        assert len(features.shape) == 4
        features = jnp.sum(
            jnp.expand_dims(features, -1) * jnp.expand_dims(kernel, 0), axis=(1, 2))
        features = jnp.reshape(features, [batch_size, -1])
        return features