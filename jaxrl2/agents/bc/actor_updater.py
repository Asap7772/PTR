from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.types import Params, PRNGKey


def log_prob_update(
        rng: PRNGKey, actor: TrainState,
        batch: FrozenDict, loss_func='log_prob') -> Tuple[PRNGKey, TrainState, Dict[str, float]]:

    rng, key = jax.random.split(rng)

    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Tuple[Any, Dict[str, float]]]:
        if actor.batch_stats is not None:
            dist, new_model_state = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats},
                              batch['observations'],
                              training=True,
                              rngs={'dropout': key},
                              mutable=['batch_stats'])
        else:
            dist = actor.apply_fn({'params': actor_params},
                batch['observations'],
                training=True,
                rngs={'dropout': key})
            new_model_state = {}
        log_probs = dist.log_prob(batch['actions'])
        log_prob_loss = -log_probs.mean()
        mse = (dist.mode() - batch['actions']) ** 2
        mse_loss = mse.mean()
        if loss_func == 'log_prob':
            actor_loss = log_prob_loss
        elif loss_func == 'mse':
            actor_loss = mse_loss
        else:
            raise ValueError
        return actor_loss, (new_model_state, {'log_probs': log_prob_loss, 'MSE': mse_loss, 'dataset_actions': batch['actions'],
                            'pred_actions_mean': dist.loc, 'action_std': dist._scale_diag})

    grads, (new_model_state, info) = jax.grad(loss_fn, has_aux=True)(actor.params)
    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)

    return rng, new_actor, info

def reward_loss_update(
        rng: PRNGKey, actor: TrainState,
        batch: FrozenDict) -> Tuple[PRNGKey, TrainState, Dict[str, float]]:

    rng, key = jax.random.split(rng)

    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Tuple[Any, Dict[str, float]]]:
        if actor.batch_stats is not None:
            dist, new_model_state = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats},
                              batch['observations'],
                              training=True,
                              rngs={'dropout': key},
                              mutable=['batch_stats'])
        else:
            dist = actor.apply_fn({'params': actor_params},
                batch['observations'],
                training=True,
                rngs={'dropout': key})
            new_model_state = {}
        pred = dist.mode().reshape(-1)
        loss = - (batch['rewards'] * jnp.log(1. / (1. + jnp.exp(-pred))) + (1.0 - batch['rewards']) * jnp.log(1. - 1. / (1. + jnp.exp(-pred))))
        # Same loss clamp trick as used in Pytorch BCELoss implementation
        loss = jax.lax.clamp(-100.0, loss, 100.0)
        # loss = (batch['rewards'] - pred) ** 2
        loss = loss.mean()
        actor_loss = loss
        return actor_loss, (new_model_state, {'reward_loss': loss, 'rewards_mean': dist.mode()})

    grads, (new_model_state, info) = jax.grad(loss_fn, has_aux=True)(actor.params)
    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)

    return rng, new_actor, info
