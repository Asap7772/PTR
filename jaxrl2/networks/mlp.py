from typing import Callable, Optional, Sequence, Union
from flax.core import frozen_dict

import numpy as np
import flax.linen as nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init


def _flatten_dict(x: Union[FrozenDict, jnp.ndarray]):
    if hasattr(x, 'values'):
        obs = []
        for k, v in sorted(x.items()):
            if k == 'state' or k == 'prev_action':
                obs.append(jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])]))
                # v = jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])])
                # pass
            else:
                obs.append(_flatten_dict(v))
        return jnp.concatenate(obs, -1)
    else:
        return x


def _flatten_dict_special(x):
    if hasattr(x, 'values'):
        obs = []
        action = None
        for k, v in sorted(x.items()):
            if k == 'state' or k == 'prev_action':
                obs.append(jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])]))
            elif k == 'actions':
                print ('action shape: ', v.shape)
                action = v
            else:
                obs.append(_flatten_dict(v))
        return jnp.concatenate(obs, -1), action
    else:
        return x


def _flatten_dict_special_v2(x, only_pixels=False):
    action = x['actions']
    print ('Action shape: ', action.shape)
    observation = x['states']

    pixels = observation['pixels']
    obs = []
    if 'state' in observation and not only_pixels:
        obs.append(
            jnp.reshape(observation['state'], [*observation['state'].shape[:-2], np.prod(observation['state'].shape[-2:])]))
    
    if 'prev_action' in observation:
        obs.append(
            jnp.reshape(observation['prev_action'], [*observation['prev_action'].shape[:-2], np.prod(observation['prev_action'].shape[-2:])]))

    if 'task_id' in observation:
        obs.append(observation['task_id'])

    if len(obs) == 0:
        return pixels, action, pixels

    return jnp.concatenate(obs, -1), action, pixels
        

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None
    init_scale: Optional[float] = 1.0
    use_normalized_features: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = _flatten_dict(x)
        print('mlp post flatten', x.shape)

        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init(self.init_scale))(x)
            
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


class MLPActionSep(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None
    init_scale: Optional[float] = 1.0
    use_normalized_features: bool = False
    use_pixel_sep: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False):
        if self.use_pixel_sep:
            x, action, pixels = _flatten_dict_special_v2(x, only_pixels=True)
            print ('Using pixel sep in MLP action sep', pixels.shape)
        else:
            x, action = _flatten_dict_special(x)
        print ('mlp action sep state post flatten', x.shape)
        print ('mlp action sep action post flatten', action.shape)

        for i, size in enumerate(self.hidden_dims):
            if self.use_pixel_sep:
                x_used = jnp.concatenate([x, action, pixels], axis=-1)
            else:
                x_used = jnp.concatenate([x, action], axis=-1)

            if i == len(self.hidden_dims)-1 and not self.activate_final:
                x = nn.Dense(size, kernel_init=default_init(1e-2))(x_used)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x_used)
            

            print ('FF layers: ', x_used.shape, x.shape)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


class MLPRepeatPerLayer(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None
    init_scale: Optional[float] = 1.
    key_for_repeat='actions'

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        repeat_tens = x[self.key_for_repeat]

        x = _flatten_dict(x)
        print('mlp post flatten', x.shape)

        for i, size in enumerate(self.hidden_dims):
            x = jnp.concatenate([x, repeat_tens], axis=-1)
            x = nn.Dense(size, kernel_init=default_init(self.init_scale))(x)
            # print('post fc size', x.shape)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


class MLPActionProject(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None
    init_scale: Optional[float] = 1.
    key_for_project='actions'
    project_size=64
    pad_size=7

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        project_tens = x[self.key_for_project]
        num_rem = self.pad_size - project_tens.shape[-1]

        npad=((0,0),) * (project_tens.ndim-1) + ((0,num_rem),)
        project_tens = jnp.pad(project_tens, npad)
        project_tens = nn.Dense(self.project_size, kernel_init=default_init(self.init_scale))(project_tens)

        x = _flatten_dict(x)
        x = jnp.concatenate([x, project_tens], axis=-1)
        print('mlp post flatten', x.shape)

        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init(self.init_scale))(x)
            # print('post fc size', x.shape)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x

class MLPProjectedRepeatPerLayer(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None
    init_scale: Optional[float] = 1.
    key_for_project='actions'
    project_size=64
    pad_size=7

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        project_tens = x[self.key_for_project]
        num_rem = self.pad_size - project_tens.shape[-1]

        npad=((0,0),) * (project_tens.ndim-1) + ((0,num_rem),)
        project_tens = jnp.pad(project_tens, npad)
        repeat_tens = nn.Dense(self.project_size, kernel_init=default_init(self.init_scale))(project_tens)

        x = _flatten_dict(x)
        print('mlp post flatten', x.shape)

        for i, size in enumerate(self.hidden_dims):
            x = jnp.concatenate([x, repeat_tens], axis=-1)
            x = nn.Dense(size, kernel_init=default_init(self.init_scale))(x)
            # print('post fc size', x.shape)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x