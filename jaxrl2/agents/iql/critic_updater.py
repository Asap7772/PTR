from typing import Dict, Tuple
from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.types import Params


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def update_v(target_critic: TrainState, value: TrainState, batch: FrozenDict,
             expectile: float,
             critic_reduction: str,
             ) -> Tuple[TrainState, Dict[str, float]]:

    if hasattr(target_critic, 'batch_stats') and target_critic.batch_stats is not None:
        qs = target_critic.apply_fn({'params': target_critic.params, 'batch_stats': target_critic.batch_stats},
                                    batch['observations'], batch['actions'], training=False, mutable=False)
    else:
        qs = target_critic.apply_fn({'params': target_critic.params}, batch['observations'], batch['actions'])

    if critic_reduction == 'min':
        q = qs.min(axis=0)
    elif critic_reduction == 'mean':
        q = qs.mean(axis=0)
    else:
        raise NotImplemented()

    def value_loss_fn(
            value_params: Params) -> Tuple[jnp.ndarray, Tuple[Any, Dict[str, float]]]:
        if value.batch_stats is not None:
            v, new_model_state = value.apply_fn({'params': value_params, 'batch_stats': value.batch_stats}, batch['observations'],
                               training=True, mutable=['batch_stats'])
        else:
            v = value.apply_fn({'params': value_params}, batch['observations'])
            new_model_state = {}
        value_loss = loss(q - v, expectile).mean()
        return value_loss, (new_model_state, {'value_loss': value_loss, 'v': v.mean()})

    grads, (new_model_state, info) = jax.grad(value_loss_fn, has_aux=True)(value.params)
    if 'batch_stats' in new_model_state:
        new_value = value.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_value = value.apply_gradients(grads=grads)

    return new_value, info

def get_stats(name, vector):
    return {name + 'mean': vector.mean(),
            name + 'min': vector.min(),
            name + 'max': vector.max(),
            }


def update_q(critic: TrainState, value: TrainState, batch: FrozenDict,
             discount: float) -> Tuple[TrainState, Dict[str, float]]:
    if value.batch_stats is not None:
        next_v = value.apply_fn({'params': value.params, 'batch_stats': value.batch_stats},
                                batch['next_observations'], training=False, mutable=False)
    else:
        next_v = value.apply_fn({'params': value.params}, batch['next_observations'])

    target_q = batch['rewards'] + discount * batch['masks'] * next_v

    def critic_loss_fn(
            critic_params: Params) -> Tuple[jnp.ndarray, Tuple[Any, Dict[str, float]]]:
        if critic.batch_stats is not None:
            qs, new_model_state = critic.apply_fn({'params': critic_params, 'batch_stats': critic.batch_stats}, batch['observations'], batch['actions'],
                                 training=True, mutable=['batch_stats'])
        else:
            qs = critic.apply_fn({'params': critic_params}, batch['observations'], batch['actions'])
            new_model_state = {}
        critic_loss = ((qs - target_q)**2).mean()
        return critic_loss, (new_model_state, {'critic_loss': critic_loss, 'q': qs.mean(), 'rewards': batch['rewards'], **get_stats('rewards', batch['rewards'])})

    grads, (new_model_state, info) = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    if 'batch_stats' in new_model_state:
        new_critic = critic.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info
