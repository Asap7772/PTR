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

def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

def compute_batch_stats_mean(batch_stats):
    flattened_batch_stats = jax.flatten_util.ravel_pytree(batch_stats)[0]
    return {
        'mean_batch_stats': jnp.mean(flattened_batch_stats),
        'min_batch_stats': jnp.min(flattened_batch_stats),
        'norm_batch_stats': jnp.linalg.norm(flattened_batch_stats),
        'batch_stats_shape': flattened_batch_stats.shape[-1]
    }

def get_cds_weights(key: PRNGKey, actor: TrainState, critic_encoder: TrainState, critic_decoder: TrainState, batch, cds_weight_temp_tau=0.995, cds_weight_temp_min=10.0, cds_weight_temp_max=10000.0, cds_weight_percentile=None, cds_weight_orig_task=False, cds_relabel_prob=0.5, true_embed_length=1024,):
    keys=jax.random.split(key, 2)
    key, keys = keys[0], keys[1:]
    
    relabel_masks=batch['relabel_masks']
    
    if hasattr(critic_encoder, 'batch_stats') and critic_encoder.batch_stats is not None:
        embed_curr_obs, _ = critic_encoder.apply_fn({'params': critic_encoder.params, 'batch_stats': critic_encoder.batch_stats}, batch['observations'], mutable=['batch_stats'], training=True)
        embed_next_obs, _ = critic_encoder.apply_fn({'params': critic_encoder.params, 'batch_stats': critic_encoder.batch_stats}, batch['next_observations'], mutable=['batch_stats'], training=True)
    else:
        embed_curr_obs = critic_encoder.apply_fn({'params': critic_encoder.params}, batch['observations'])
        embed_next_obs = critic_encoder.apply_fn({'params': critic_encoder.params}, batch['next_observations'])
    
    if hasattr(critic_decoder, 'batch_stats') and critic_decoder.batch_stats is not None:
        qs, _ = critic_decoder.apply_fn({'params': critic_decoder.params, 'batch_stats': critic_decoder.batch_stats},embed_curr_obs, batch['actions'], mutable=['batch_stats'], rngs={'dropout': keys[0]}, training=True)
    else:
        qs = critic_decoder.apply_fn( {'params': critic_decoder.params}, embed_curr_obs, batch['actions'], rngs={'dropout': keys[0]})
    
    qs_orig = torch.masked_select(qs, relabel_masks == 0.0) # TODO: find replacement for this
    
    if cds_weight_percentile is not None:
        qs_orig_avg = jnp.percentile(qs_orig, cds_weight_percentile, axis=0, interpolation='nearest', keepdims=True).squeeze()
    else:
        qs_orig_avg = qs_orig.mean()
        
        
    q_diff = qs - qs_orig_avg

    if cds_weight_temp_value is None:
        cds_weight_temp_value = q_diff.abs().mean()
    else:
        cds_weight_temp_value = cds_weight_temp_tau * cds_weight_temp_value + (1.0 - cds_weight_temp_tau) * q_diff.abs().mean()
    
                                    
    cds_weight_temp_avg = jnp.clamp(cds_weight_temp_value, min=cds_weight_temp_min, max=cds_weight_temp_max)
    q_diff_normalized = q_diff / cds_weight_temp_avg
    
    cds_weights = (sigmoid(q_diff_normalized) * relabel_masks + (1.0 - relabel_masks)).detach()
    
    cds_weight1, cds_weight2 = cds_weights[0], cds_weights[1]
    
    cds_q1_entropy = cds_weight1.detach() * jnp.log(cds_weight1.detach())
    cds_q1_entropy = cds_q1_entropy.sum()

    cds_q2_entropy = cds_weight2.detach() * jnp.log(cds_weight2.detach())
    cds_q2_entropy = cds_q2_entropy.sum()

    cds_weight = jnp.min(cds_weight1, cds_weight2)
    
    return cds_weight, cds_weight1, cds_weight2, cds_q1_entropy, cds_q2_entropy


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
        

def get_adversarial_loss(q_aux, q_base, next_q_aux, next_q_base, adv_loss_version: int = 0,
                         adv_loss_coefficient=0.0):
    mv_norm = jnp.mean(jnp.abs(q_base), axis=-1)
    nv_norm = jnp.mean(jnp.abs(next_q_base), axis=-1)
    eps_threshold = 1.0
    
    mv_sq_for_aux = jnp.mean(q_aux ** 2, axis=-1)
    mv_sq_for_base = jnp.mean(q_base ** 2, axis=-1)
    nv_sq_for_base = jnp.mean(next_q_base ** 2, axis=-1)
    nv_sq_for_aux = jnp.mean(next_q_aux ** 2, axis=-1)

    adv_loss_for_aux = jnp.mean(mv_sq_for_aux) - jnp.mean(nv_sq_for_aux)
    if adv_loss_version == 0:
        adv_loss_for_base = jnp.mean(jax.nn.sigmoid(eps_threshold - mv_norm) * nv_norm)
    elif adv_loss_version == 1:
        adv_loss_for_base = jnp.mean(nv_sq_for_base) - jnp.mean(mv_sq_for_base)
    else:
        adv_loss_for_base = jnp.mean(nv_sq_for_base) - jax.lax.stop_gradient(
            jnp.mean(mv_sq_for_base))
    
    ret_dict = dict()
    ret_dict['sanity_check'] = adv_loss_for_aux + adv_loss_for_base
    ret_dict['adv_loss_for_adversary'] = adv_loss_for_aux
    ret_dict['adv_loss_for_base'] = adv_loss_for_base
    return (adv_loss_for_aux + adv_loss_coefficient * adv_loss_for_base), ret_dict


def update_critic(
        key: PRNGKey, actor: TrainState, critic_encoder: TrainState, critic_decoder: TrainState,
        target_critic_encoder: TrainState, target_critic_decoder: TrainState, temp: TrainState, batch: DatasetDict,
        discount: float, backup_entropy: bool, critic_reduction: str, cql_alpha: float, max_q_backup: bool, dr3_coefficient: float,
        method:bool=False, method_const:float=1.0, method_type:int=0, cross_norm:bool=False,
        use_basis_projection:bool=False, basis_projection_coefficient:float=0.0,
        use_gaussian_policy: bool = False, min_q_version: int = 3,
    ):

    key, key_pi, key_random, key_temp, key_q = jax.random.split(key, num=5)
    key_q1, key_q2, key_q3, key_q4, key_q5 = jax.random.split(key_q, num=5)

    print ('Using min q version in critic updater: ', min_q_version)
    
    if hasattr(target_critic_encoder, 'batch_stats') and target_critic_encoder.batch_stats is not None:
        embed_next_obs, _ = target_critic_encoder.apply_fn(
            {'params': target_critic_encoder.params, 'batch_stats': target_critic_encoder.batch_stats}, 
            batch['next_observations'], training=False, mutable=['batch_stats'])
    else:
        embed_next_obs = target_critic_encoder.apply_fn({'params': target_critic_encoder.params}, batch['next_observations'])
    
    if max_q_backup:
        # needed for actor
        next_observations_tiled = extend_and_repeat(
            batch['next_observations'], axis=1, repeat=NUM_CQL_REPEAT
        )
        next_observations_tiled = reshape_for_cql_computation(
            next_observations_tiled, num_cql_repeat=NUM_CQL_REPEAT)
        
        #embedding tiled
        embed_next_obs = extend_and_repeat(
            embed_next_obs, axis=1, repeat=NUM_CQL_REPEAT
        )
        embed_next_obs = reshape_for_cql_computation(
            embed_next_obs, num_cql_repeat=NUM_CQL_REPEAT)
    else:
        next_observations_tiled = batch['next_observations']

    if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
        dist, _ = actor.apply_fn({'params': actor.params, 'batch_stats': actor.batch_stats}, 
                                 next_observations_tiled, mutable=['batch_stats'],  training=False)
    else:
        dist = actor.apply_fn({'params': actor.params}, next_observations_tiled)
        
    if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
        bp_dist, _ = actor.apply_fn({'params': actor.params, 'batch_stats': actor.batch_stats},
                                    batch['next_observations'], mutable=['batch_stats'],  training=False)
    else:
        bp_dist = actor.apply_fn({'params': actor.params}, batch['next_observations'])

    key, bp_key = jax.random.split(key)
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    bp_next_actions, _ = bp_dist.sample_and_log_prob(seed=bp_key)

    if hasattr(target_critic_decoder, 'batch_stats') and target_critic_decoder.batch_stats is not None:
        next_qs, _ = target_critic_decoder.apply_fn(
            {'params': target_critic_decoder.params, 'batch_stats': target_critic_decoder.batch_stats},
            embed_next_obs, next_actions, mutable=['batch_stats'], training=False,
            rngs={'dropout': key_q1})
    else:
        next_qs = target_critic_decoder.apply_fn(
            {'params': target_critic_decoder.params}, embed_next_obs, next_actions,
            rngs={'dropout': key_q1})
    
    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplemented()

    if max_q_backup:
        """Now reduce next q over the actions"""
        next_q_reshape = jnp.reshape(next_q, (batch['actions'].shape[0], NUM_CQL_REPEAT))
        next_q = jnp.max(next_q_reshape, axis=-1)

    target_q = batch['rewards'] + discount * batch['masks'] * next_q

    if backup_entropy:
        target_q -= discount * batch['masks'] * temp.apply_fn(
            {'params': temp.params}) * next_log_probs


    # CQL sample actions
    observations_tiled = extend_and_repeat(batch['observations'], axis=1, repeat=NUM_CQL_REPEAT)
    observations_tiled = reshape_for_cql_computation(
        observations_tiled, num_cql_repeat=NUM_CQL_REPEAT)

    next_observations_tiled_temp = extend_and_repeat(batch['next_observations'], axis=1, repeat=NUM_CQL_REPEAT)
    next_observations_tiled_temp = reshape_for_cql_computation(
        next_observations_tiled_temp, num_cql_repeat=NUM_CQL_REPEAT)
    
    if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
        policy_dist, _ = actor.apply_fn({'params': actor.params, 'batch_stats': actor.batch_stats},
                                        observations_tiled, mutable=['batch_stats'],  training=False)
    else:
        policy_dist = actor.apply_fn({'params': actor.params}, observations_tiled)
    
    policy_actions, policy_log_probs = policy_dist.sample_and_log_prob(seed=key_pi)
    
    if isinstance(batch['observations'], frozen_dict.FrozenDict):
        n = batch['observations']['pixels'].shape[0]
    else:
        n = batch['observations'].shape[0]

    if not use_gaussian_policy:
        random_actions = jax.random.uniform(
            key_random, shape=(n * NUM_CQL_REPEAT, policy_actions.shape[-1]),
            minval=-1.0, maxval=1.0
        )
        random_pi = (1.0/2.0) ** policy_actions.shape[-1]
    else:
        random_actions = jax.random.uniform(
            key_random, shape=(n * NUM_CQL_REPEAT, policy_actions.shape[-1]),
            minval=-3.0, maxval=3.0
        )
        random_pi = (1.0/6.0) ** policy_actions.shape[-1]


    def critic_loss_fn(critic_encoder_params: Params, critic_decoder_params: Params):
        if hasattr(critic_encoder, 'batch_stats') and critic_encoder.batch_stats is not None:
            embed_curr_obs, new_model_state_encoder = critic_encoder.apply_fn(
                {'params': critic_encoder_params, 'batch_stats': critic_encoder.batch_stats}, 
                batch['observations'], mutable=['batch_stats'], training=True)
            embed_next_obs_main, _ = critic_encoder.apply_fn(
                {'params': critic_encoder_params, 'batch_stats': critic_encoder.batch_stats},
                batch['next_observations'], mutable=['batch_stats'], training=True)
        else:
            embed_curr_obs = critic_encoder.apply_fn({'params': critic_encoder_params}, batch['observations'])
            new_model_state_encoder = {}
            embed_next_obs_main = critic_encoder.apply_fn(
                {'params': critic_encoder_params}, batch['next_observations'])
            
        embed_curr_obs_tiled = extend_and_repeat(embed_curr_obs, axis=1, repeat=NUM_CQL_REPEAT)
        embed_curr_obs_tiled = reshape_for_cql_computation(embed_curr_obs_tiled, num_cql_repeat=NUM_CQL_REPEAT)
        
        if hasattr(critic_decoder, 'batch_stats') and critic_decoder.batch_stats is not None:
            qs, new_model_state_decoder = critic_decoder.apply_fn(
                {'params': critic_decoder_params, 'batch_stats': critic_decoder.batch_stats},
                embed_curr_obs, batch['actions'], mutable=['batch_stats'],
                rngs={'dropout': key_q2}, training=True)
        else:
            qs = critic_decoder.apply_fn(
                {'params': critic_decoder_params}, embed_curr_obs, batch['actions'],
                rngs={'dropout': key_q2})
            new_model_state_decoder = {}
            
            
        new_model_state = (new_model_state_encoder, new_model_state_decoder)

        critic_loss = ((qs - target_q)**2).mean()
        bellman_loss = critic_loss

        qs_to_log = copy.deepcopy(qs) # copy to avoid modifying the original qs
        
        if hasattr(critic_decoder, 'batch_stats') and critic_decoder.batch_stats is not None:
            q_pi, _ = critic_decoder.apply_fn(
                {'params': critic_decoder_params, 'batch_stats': critic_decoder.batch_stats},
                embed_curr_obs_tiled, policy_actions, mutable=['batch_stats'], training=True,
                rngs={'dropout': key_q3})
        else:
            q_pi = critic_decoder.apply_fn(
                {'params': critic_decoder_params}, embed_curr_obs_tiled, policy_actions,
                rngs={'dropout': key_q3})
        
        q_pi_for_is = (q_pi[0] - policy_log_probs, q_pi[1] - policy_log_probs)

        # When not using importance sampling, we can use this version
        q_pi_for_minq_v2 = (
            jnp.reshape(q_pi[0], (batch['actions'].shape[0], NUM_CQL_REPEAT)),
            jnp.reshape(q_pi[1], (batch['actions'].shape[0], NUM_CQL_REPEAT))
        )
        q_pi_for_minq_v2 = jnp.stack(q_pi_for_minq_v2, axis=0)

        q_pi_for_is = (
            jnp.reshape(q_pi_for_is[0], (batch['actions'].shape[0], NUM_CQL_REPEAT)),
            jnp.reshape(q_pi_for_is[1], (batch['actions'].shape[0], NUM_CQL_REPEAT))
        )
        q_pi_for_is = jnp.stack(q_pi_for_is, axis=0)

        if hasattr(critic_decoder, 'batch_stats') and critic_decoder.batch_stats is not None:
            q_random, _ = critic_decoder.apply_fn(
                {'params': critic_decoder_params, 'batch_stats': critic_decoder.batch_stats},
                embed_curr_obs_tiled, random_actions, mutable=['batch_stats'], training=True,
                rngs={'dropout': key_q4})
        else:
            q_random = critic_decoder.apply_fn(
                {'params': critic_decoder_params}, embed_curr_obs_tiled, random_actions,
                rngs={'dropout': key_q4})

        q_random_for_is = (q_random[0] - np.log(random_pi), q_random[1] - np.log(random_pi))
        q_random_for_is = (
            jnp.reshape(q_random_for_is[0], (batch['actions'].shape[0], NUM_CQL_REPEAT)),
            jnp.reshape(q_random_for_is[1], (batch['actions'].shape[0], NUM_CQL_REPEAT))
        )    
        q_random_for_is = jnp.stack(q_random_for_is, axis=0)

        q_random_for_minq_v2 = (
            jnp.reshape(q_random[0], (batch['actions'].shape[0], NUM_CQL_REPEAT)),
            jnp.reshape(q_random[1], (batch['actions'].shape[0], NUM_CQL_REPEAT))
        )
        q_random_for_minq_v2 = jnp.stack(q_random_for_minq_v2, axis=0)

        if min_q_version == 1:
            cat_q = q_pi_for_minq_v2
            lse_q = jnp.mean(cat_q, axis=-1)
        elif min_q_version == 2:
            cat_q = jnp.concatenate([q_pi_for_minq_v2, q_random_for_minq_v2], axis=-1)
            lse_q = jax.scipy.special.logsumexp(cat_q, axis=-1)
        else:
            cat_q = jnp.concatenate([q_pi_for_is, q_random_for_is], axis=-1)
            lse_q = jax.scipy.special.logsumexp(cat_q, axis=-1)
        
        cql_loss_per_element = lse_q - qs

        cql_loss = cql_loss_per_element.mean()

        critic_loss = critic_loss + cql_alpha * cql_loss
        
        # DR3 loss
        if dr3_coefficient >= 0.0:
            dr3_loss = jnp.mean(batch['masks'] * jnp.sum(embed_curr_obs['pixels'] * embed_next_obs_main['pixels'], axis=-1))
            critic_loss = critic_loss + dr3_coefficient * dr3_loss
            pixel_norm_current = jnp.linalg.norm(embed_curr_obs['pixels'], axis=-1).mean()
            pixel_norm_next = jnp.linalg.norm(embed_next_obs_main['pixels'], axis=-1).mean()

        if use_basis_projection:      
            # TODO(aviralkumar): This is not compatible with dropout yet, need to add keys
            if hasattr(critic_decoder, 'batch_stats') and critic_decoder.batch_stats is not None:
                qs_for_aux, _ = critic_decoder.apply_fn(
                    {'params': critic_decoder_params, 'batch_stats': critic_decoder.batch_stats},
                    embed_curr_obs, batch['actions'], version_type=1, mutable=['batch_stats'], training=True)
                qs_for_base, _ = critic_decoder.apply_fn(
                    {'params': critic_decoder_params, 'batch_stats': critic_decoder.batch_stats},
                    embed_curr_obs, batch['actions'], version_type=2, mutable=['batch_stats'], training=True)
                next_qs_for_aux, _ = critic_decoder.apply_fn(
                    {'params': critic_decoder_params, 'batch_stats': critic_decoder.batch_stats},
                    embed_next_obs_main, bp_next_actions, version_type=1, mutable=['batch_stats'], training=True)
                next_qs_for_base, _ = critic_decoder.apply_fn(
                    {'params': critic_decoder_params, 'batch_stats': critic_decoder.batch_stats},
                    embed_next_obs_main, bp_next_actions, version_type=2, mutable=['batch_stats'], training=True)
            else:
                qs_for_aux = critic_decoder.apply_fn({'params': critic_decoder_params},
                                                     embed_curr_obs, batch['actions'], training=True, version_type=1)
                qs_for_base = critic_decoder.apply_fn({'params': critic_decoder_params},
                                                      embed_curr_obs, batch['actions'], training=True, version_type=2)
                next_qs_for_aux = critic_decoder.apply_fn({'params': critic_decoder_params},
                                                          embed_next_obs_main, bp_next_actions, training=True, version_type=1)
                next_qs_for_base = critic_decoder.apply_fn({'params': critic_decoder_params},
                                                           embed_next_obs_main, bp_next_actions, training=True, version_type=2)


            adversarial_loss, adv_dict = get_adversarial_loss(
                q_aux=qs_for_aux,
                q_base=qs_for_base,
                next_q_aux=next_qs_for_aux,
                next_q_base=next_qs_for_base,
                adv_loss_version=2,
                adv_loss_coefficient=basis_projection_coefficient)

            critic_loss += adversarial_loss

        ## Logging only
        diff_rand_data = q_random.mean() - qs_to_log.mean()
        diff_pi_data = q_pi.mean() - qs_to_log.mean()

        things_to_log = {   
            'critic_loss': critic_loss,
            'bellman_loss': bellman_loss,
            'cql_loss_mean': cql_loss,
            'cql_loss_max': jnp.max(cql_loss_per_element),
            'cql_loss_min': jnp.min(cql_loss_per_element),
            'cql_loss_std': jnp.std(cql_loss_per_element),
            'lse_q': lse_q.mean(),
            'q_pi_avg': q_pi.mean(),
            'q_random': q_random.mean(),
            'q_data_avg': qs_to_log.mean(),
            'q_data_max': qs_to_log.max(),
            'q_data_min': qs_to_log.min(),
            'q_data_std': qs_to_log.std(),
            'weighted_q_data_avg': qs.mean(),
            'weighted_q_data_max': qs.max(),
            'weighted_q_data_min': qs.min(),
            'weighted_q_data_std': qs.std(),
            'q_pi_max': q_pi.max(),
            'q_pi_min': q_pi.min(),
            'diff_pi_data_mean': diff_pi_data,
            'diff_rand_data_mean': diff_rand_data, 
            'target_actor_entropy': -next_log_probs.mean(),
            'rewards_mean': batch['rewards'].mean(),
            'actions_mean': batch['actions'].mean(),
            'actions_max': batch['actions'].max(),
            'actions_min': batch['actions'].min(),
            'terminals_mean': batch['masks'].mean(),
            'log_pis_mean': policy_log_probs.mean(),
            'log_pis_max': policy_log_probs.max(),
            'log_pis_min': policy_log_probs.min(),
        }
        if dr3_coefficient >= 0.0:
            things_to_log['dr3_loss'] = dr3_loss
            things_to_log['pixel_norm_current'] = pixel_norm_current
            things_to_log['pixel_norm_next'] = pixel_norm_next
        
        if use_basis_projection:
            things_to_log.update(adv_dict)
            
        if hasattr(critic_encoder, 'batch_stats') and critic_encoder.batch_stats is not None:
            batch_stats_dict = compute_batch_stats_mean(new_model_state)
            things_to_log.update(batch_stats_dict)

        return critic_loss, (things_to_log, new_model_state)

    (grads_encoder,grads_decoder), (info,new_model_state) = jax.grad(critic_loss_fn, has_aux=True, argnums=(0,1))(critic_encoder.params, critic_decoder.params)
    
    grads_encoder = jax.lax.pmean(grads_encoder, axis_name='pmap')
    grads_decoder = jax.lax.pmean(grads_decoder, axis_name='pmap')
    info = jax.lax.pmean(info, axis_name='pmap')
    # new_model_state = jax.lax.pmean(new_model_state, axis_name='pmap')
    
    if 'batch_stats' in new_model_state[0]:
        new_critic_encoder = critic_encoder.apply_gradients(grads=grads_encoder, batch_stats=new_model_state[0]['batch_stats'])
        print ('updated batch stats for encoder')
    else:
        new_critic_encoder = critic_encoder.apply_gradients(grads=grads_encoder)
        
    if 'batch_stats' in new_model_state[1]:
        new_critic_decoder = critic_decoder.apply_gradients(grads=grads_decoder, batch_stats=new_model_state[1]['batch_stats'])
        print ('updated batch stats for decoder')
    else:
        new_critic_decoder = critic_decoder.apply_gradients(grads=grads_decoder)

    new_critic = (new_critic_encoder, new_critic_decoder)
    return new_critic, info