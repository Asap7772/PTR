from audioop import cross
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


def compute_batch_stats_mean(batch_stats):
    flattened_batch_stats = jax.flatten_util.ravel_pytree(batch_stats)[0]
    return {
        'actor_mean_batch_stats': jnp.mean(flattened_batch_stats),
        'actor_min_batch_stats': jnp.min(flattened_batch_stats),
        'actor_norm_batch_stats': jnp.linalg.norm(flattened_batch_stats),
        'actor_batch_stats_shape': flattened_batch_stats.shape[-1]
    }

def update_actor(key: PRNGKey, actor: TrainState, critic_encoder: TrainState, critic_decoder: TrainState,
                 temp: TrainState, batch: DatasetDict, cross_norm:bool=False,
                 use_gaussian_policy: bool = False, autoregressive_policy: bool = False):
    
    key, key_act, key_q = jax.random.split(key, num=3)

    def actor_loss_fn(actor_params: Params):
        
        if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
            dist, new_model_state = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, 
                                                   batch['observations'], mutable=['batch_stats'],  training=True)
        else:
            dist = actor.apply_fn({'params': actor_params}, batch['observations'])
            new_model_state = {}

        if not use_gaussian_policy and not autoregressive_policy:
            mean_dist = dist.distribution._loc
            std_diag_dist = dist.distribution._scale_diag
        else:
            mean_dist = dist._loc
            std_diag_dist = dist._scale_diag

        mean_dist_norm = jnp.linalg.norm(mean_dist, axis=-1)
        std_dist_norm = jnp.linalg.norm(std_diag_dist, axis=-1)

        if not use_gaussian_policy:
            actions, log_probs = dist.sample_and_log_prob(seed=key_act)
        else:
            actions = mean_dist
            log_probs = jnp.zeros_like(actions)

        if not use_gaussian_policy:
            dataset_actions_log_prob = dist.log_prob(batch['actions'])
            dataset_actions_mse = jnp.sum((batch['actions'] - jnp.tanh(mean_dist))**2, axis=-1)
        else:
            dataset_actions_log_prob = jnp.sum((batch['actions'] - actions)**2, axis=-1)
            dataset_actions_mse = dist.log_prob(batch['actions'])

        if hasattr(critic_encoder, 'batch_stats') and critic_encoder.batch_stats is not None:
            embed_curr_obs, _ = critic_encoder.apply_fn({'params': critic_encoder.params, 'batch_stats': critic_encoder.batch_stats}, batch['observations'], mutable=['batch_stats'], training=False)
        else:
            embed_curr_obs = critic_encoder.apply_fn({'params': critic_encoder.params}, batch['observations'])
            
        if hasattr(critic_decoder, 'batch_stats') and critic_decoder.batch_stats is not None:
            qs, _ = critic_decoder.apply_fn({'params': critic_decoder.params, 'batch_stats': critic_decoder.batch_stats},
                                            embed_curr_obs, actions, mutable=['batch_stats'],  training=False,
                                            rngs={'dropout': key_q})
        else:
            qs = critic_decoder.apply_fn(
                {'params': critic_decoder.params}, embed_curr_obs, actions,
                rngs={'dropout': key_q})
        
        q = qs.min(axis=0)
        
        if not use_gaussian_policy:
            actor_loss = (log_probs * temp.apply_fn({'params': temp.params}) - q).mean()
            if autoregressive_policy:
                pred_actions_mean = jnp.tanh(dist._loc)
            else:
                pred_actions_mean = jnp.tanh(dist.distribution._loc)
        else:
            actor_loss = -(q.mean())
            pred_actions_mean = dist._loc

        things_to_log = {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'q_pi_in_actor': q.mean(),
            'mean_pi_norm': mean_dist_norm.mean(),
            'std_pi_norm': std_dist_norm.mean(),
            'mean_pi_avg': mean_dist.mean(),
            'mean_pi_max': mean_dist.max(),
            'mean_pi_min': mean_dist.min(),
            'std_pi_avg': std_diag_dist.mean(),
            'std_pi_max': std_diag_dist.max(),
            'std_pi_min': std_diag_dist.min(),
            'pred_actions_mean': pred_actions_mean,
            'dataset_actions_log_prob_mean': dataset_actions_log_prob.mean(),
            'dataset_actions_log_prob_min': dataset_actions_log_prob.min(),
            'dataset_actions_log_prob_max': dataset_actions_log_prob.max(),
            'dataset_actions_log_prob_std': jnp.std(dataset_actions_log_prob),
            'dataset_actions': batch['actions']
        }
        if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
            batch_stats_dict = compute_batch_stats_mean(new_model_state)
            things_to_log.update(batch_stats_dict)
            
        return actor_loss, (things_to_log, new_model_state)

    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    
    grads = jax.lax.pmean(grads, axis_name='pmap')
    info = jax.lax.pmean(info, axis_name='pmap')
    # new_model_state = jax.lax.pmean(new_model_state, axis_name='pmap')
    
    if 'batch_stats' in new_model_state:
        print ('Applying batch norm in actor')
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info