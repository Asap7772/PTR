from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.types import Params, PRNGKey


def log_prob_update_bc(
        rng: PRNGKey, actor: TrainState, critic_encoder: TrainState, critic_decoder: TrainState,
        batch: FrozenDict, temperature=1.0) -> Tuple[PRNGKey, TrainState, Dict[str, float]]:

    rng, key, key_encoder, key_decoder1, key_decoder2 , key_sample, key_act = jax.random.split(rng, 7)

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
        
        if critic_encoder.batch_stats is not None:
            obs_embed = critic_encoder.apply_fn({'params': critic_encoder.params, 'batch_stats': critic_encoder.batch_stats}, batch['observations'], training=True, rngs={'dropout': key_encoder})
        else:
            obs_embed = critic_encoder.apply_fn({'params': critic_encoder.params}, batch['observations'], training=True, rngs={'dropout': key_encoder})
            
        if critic_decoder.batch_stats is not None:
            q_data = critic_decoder.apply_fn({'params': critic_decoder.params, 'batch_stats': critic_decoder.batch_stats}, obs_embed, batch['actions'], training=True, rngs={'dropout': key_decoder1})
            q_pi = critic_decoder.apply_fn({'params': critic_decoder.params, 'batch_stats': critic_decoder.batch_stats}, obs_embed, action_pi, training=True, rngs={'dropout': key_decoder2})
        else:
            q_data = critic_decoder.apply_fn({'params': critic_decoder.params}, obs_embed, batch['actions'], training=True, rngs={'dropout': key_decoder1})
            q_pi = critic_decoder.apply_fn({'params': critic_decoder.params}, obs_embed, action_pi, training=True, rngs={'dropout': key_decoder2})
            
        advantage = q_data-q_pi
        clipped_advantage = jnp.clip(advantage/temperature, a_max=5)
        actor_loss = -(log_probs*jnp.exp(clipped_advantage)).mean()

        # sample log pis for entropy calculation
        _, entropy = dist.sample_and_log_prob(seed=key_act)
        
        things_to_log={
            'advantages_mean': advantage.mean(), 
            'advantages_std': advantage.std(),
            'advantages_max': advantage.max(), 
            'advantages_min': advantage.min(),
            'clipped_advantages_mean': clipped_advantage.mean(), 
            'clipped_advantages_std': clipped_advantage.std(),
            'clipped_advantages_max': clipped_advantage.max(), 
            'clipped_advantages_min': clipped_advantage.min(),
            'log_probs': log_prob_loss,
            'MSE': mse_loss, 
            'dataset_actions': batch['actions'],
            'pred_actions_mean': dist.distribution.loc, 
            'action_std': dist.distribution._scale_diag,
            'loss': actor_loss,
            'entropy': -entropy.mean(),
        }

        things_to_log = {'bc_' + k: v for k, v in things_to_log.items()}

        actor_loss = mse_loss # just do mse for bc phase but log awbc loss above

        return actor_loss, (new_model_state, things_to_log)

    grads, (new_model_state, info) = jax.grad(loss_fn, has_aux=True)(actor.params)
    
    grads = jax.lax.pmean(grads, axis_name='pmap')
    info = jax.lax.pmean(info, axis_name='pmap')
    
    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info