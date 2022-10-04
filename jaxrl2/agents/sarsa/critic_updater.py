from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey

from flax.core import frozen_dict
import copy

NUM_CQL_REPEAT = 4
CLIP_MIN=0
CLIP_MAX=20


def extend_and_repeat(tensor, axis, repeat):
    if isinstance(tensor, frozen_dict.FrozenDict):
        new_tensor = {}
        for key in tensor:
            new_tensor[key] = jnp.repeat(jnp.expand_dims(tensor[key], axis), repeat, axis=axis)
        new_tensor = tensor.copy(add_or_replace=new_tensor)
        return new_tensor
    else:
        return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)


def reshape_for_cql_computation(tensor, num_cql_repeat):
    if isinstance(tensor, frozen_dict.FrozenDict):
        new_tensor = {}
        for key in tensor:
            new_tensor[key] = jnp.reshape(tensor[key],
                    [tensor[key].shape[0] * num_cql_repeat, *tensor[key].shape[2:]])
        new_tensor = tensor.copy(add_or_replace=new_tensor)
        return new_tensor
    else:
        return jnp.reshape(tensor,
                [tensor.shape[0] * num_cql_repeat, *tensor.shape[2:]])

def update_critic(
        key: PRNGKey, 
        actor: TrainState,
        critic: TrainState, 
        target_critic: TrainState, 
        temp: TrainState, 
        batch: DatasetDict,
        discount: float, 
        backup_entropy: bool, 
        critic_reduction: str, 
        cql_alpha: float, 
        max_q_backup: bool, 
        dr3_coefficient: float,
        method:bool=False, 
        method_const:float=0,
        method_type:int=0, 
        method_alpha=0.5,
        cross_norm=False,
    ) -> Tuple[TrainState, Dict[str, float]]:

    key, key_pi, key_random = jax.random.split(key, num=3)
    next_observations_tiled = batch['next_observations']
    next_actions=batch['next_actions']
    
    if hasattr(target_critic, 'batch_stats') and target_critic.batch_stats is not None:
        next_qs = target_critic.apply_fn({'params': target_critic.params, 'batch_stats': target_critic.batch_stats}, next_observations_tiled, next_actions)
    else:
        next_qs = target_critic.apply_fn({'params': target_critic.params}, next_observations_tiled, next_actions)
    next_q = next_qs.mean(axis=0)

    target_q = batch['rewards'] + discount * batch['masks'] * next_q


    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        if hasattr(critic, 'batch_stats') and critic.batch_stats is not None:
            qs, new_model_state = critic.apply_fn({'params': critic_params, 'batch_stats': critic.batch_stats}, batch['observations'], batch['actions'], mutable=['batch_stats'])
        else:
            qs = critic.apply_fn({'params': critic_params}, batch['observations'], batch['actions'])
            new_model_state = {}
            
        critic_loss = ((qs - target_q)**2).mean()
        bellman_loss = critic_loss

        qs_to_log = copy.deepcopy(qs) # copy to avoid modifying the original qs

        things_to_log = {   
            'critic_loss': critic_loss,
            'bellman_loss': bellman_loss,
            'q_data_avg': qs_to_log.mean(),
            'q_data_max': qs_to_log.max(),
            'q_data_min': qs_to_log.min(),
            'q_data_std': qs_to_log.std(),
            'weighted_q_data_avg': qs.mean(),
            'weighted_q_data_max': qs.max(),
            'weighted_q_data_min': qs.min(),
            'weighted_q_data_std': qs.std(),
            'rewards_mean': batch['rewards'].mean(),
            'actions_mean': batch['actions'].mean(),
            'actions_max': batch['actions'].max(),
            'actions_min': batch['actions'].min(),
            'next_actions_mean': batch['next_actions'].mean(),
            'next_actions_max': batch['next_actions'].max(),
            'next_actions_min': batch['next_actions'].min(),
            'terminals_mean': batch['masks'].mean(),
        }
        return critic_loss, (things_to_log, new_model_state)

    grads, (info, new_model_state) = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    
    if 'batch_stats' in new_model_state:
        new_critic = critic.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info