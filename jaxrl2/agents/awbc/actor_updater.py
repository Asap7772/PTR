from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.types import Params, PRNGKey


def log_prob_update(
        rng: PRNGKey, actor: TrainState, critic_encoder: TrainState, critic_decoder: TrainState,
        batch: FrozenDict, loss_func='log_prob', temperature=1.0) -> Tuple[PRNGKey, TrainState, Dict[str, float]]:

    rng, key, key_encoder, key_decoder1, key_decoder2 , key_sample = jax.random.split(rng, 6)

    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Tuple[Any, Dict[str, float]]]:
        if actor.batch_stats is not None:
            dist, new_model_state = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats},
                              batch['observations'], training=True, rngs={'dropout': key}, mutable=['batch_stats'])
        else:
            dist = actor.apply_fn({'params': actor_params}, batch['observations'], training=True, rngs={'dropout': key})
            new_model_state = {}
            
        log_probs = dist.log_prob(batch['actions'])
        log_prob_loss = - log_probs.mean()
        mse = (dist.mode() - batch['actions']) ** 2
        mse_loss = mse.mean() 
        
        action_pi, log_pi = dist.sample_and_log_prob(seed=key)
        
        if hasattr(critic_encoder, 'batch_stats') and critic_encoder.batch_stats is not None:
            obs_embed = critic_encoder.apply_fn({'params': critic_encoder.params, 'batch_stats': critic_encoder.batch_stats}, batch['observations'], training=True, rngs={'dropout': key_encoder})
        else:
            obs_embed = critic_encoder.apply_fn({'params': critic_encoder.params}, batch['observations'], training=True, rngs={'dropout': key_encoder})
            
        if hasattr(critic_decoder, 'batch_stats') and critic_decoder.batch_stats is not None:
            q_data = critic_decoder.apply_fn({'params': critic_decoder.params, 'batch_stats': critic_decoder.batch_stats}, obs_embed, batch['actions'], training=True, rngs={'dropout': key_decoder1})
            q_pi = critic_decoder.apply_fn({'params': critic_decoder.params, 'batch_stats': critic_decoder.batch_stats}, obs_embed, action_pi, training=True, rngs={'dropout': key_decoder2})
        else:
            q_data = critic_decoder.apply_fn({'params': critic_decoder.params}, obs_embed, batch['actions'], training=True, rngs={'dropout': key_decoder1})
            q_pi = critic_decoder.apply_fn({'params': critic_decoder.params}, obs_embed, action_pi, training=True, rngs={'dropout': key_decoder2})
            
        advantage = q_data-q_pi
        actor_loss = -(log_probs*jnp.exp(1/temperature * advantage)).mean()
        
        return actor_loss, (new_model_state, {'log_probs': log_prob_loss, 'MSE': mse_loss, 'dataset_actions': batch['actions'], 'pred_actions_mean': dist.loc, 'action_std': dist._scale_diag})

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
