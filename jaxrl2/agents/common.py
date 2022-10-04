from functools import partial
from typing import Callable, Tuple, Any

import distrax
import jax
import jax.numpy as jnp
import numpy as np

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey

# Helps to minimize CPU to GPU transfer.
def _unpack(batch):
    # Assuming that if next_observation is missing, it's combined with observation:
    obs_pixels = batch['observations']['pixels'][..., :-1]
    next_obs_pixels = batch['observations']['pixels'][..., 1:]

    obs = batch['observations'].copy(add_or_replace={'pixels': obs_pixels})
    next_obs = batch['next_observations'].copy(
        add_or_replace={'pixels': next_obs_pixels})

    batch = batch.copy(add_or_replace={
        'observations': obs,
        'next_observations': next_obs
    })

    return batch

@partial(jax.jit, static_argnames='actor_apply_fn')
def eval_log_prob_jit(actor_apply_fn: Callable[..., distrax.Distribution],
                      actor_params: Params, actor_batch_stats: Any, batch: DatasetDict) -> float:
    batch = _unpack(batch)
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist = actor_apply_fn(input_collections, 
                          batch['observations'],
                          training=False,
                          mutable=False)
    log_probs = dist.log_prob(batch['actions'])
    return log_probs.mean()

@partial(jax.jit, static_argnames='actor_apply_fn')
def eval_mse_jit(actor_apply_fn: Callable[..., distrax.Distribution],
                      actor_params: Params, actor_batch_stats: Any, batch: DatasetDict) -> float:
    batch = _unpack(batch)
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist = actor_apply_fn(input_collections, 
                          batch['observations'],
                          training=False,
                          mutable=False)
    mse = (dist.loc - batch['actions']) ** 2
    return mse.mean()

def eval_reward_function_jit(actor_apply_fn: Callable[..., distrax.Distribution],
                      actor_params: Params, actor_batch_stats: Any, batch: DatasetDict) -> float:
    batch = _unpack(batch)
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist = actor_apply_fn(input_collections, 
                          batch['observations'],
                          training=False,
                          mutable=False)
    pred = dist.mode().reshape(-1)
    loss = - (batch['rewards'] * jnp.log(1. / (1. + jnp.exp(-pred))) + (1.0 - batch['rewards']) * jnp.log(1. - 1. / (1. + jnp.exp(-pred))))
    return loss.mean()


@partial(jax.jit, static_argnames='actor_apply_fn')
def eval_actions_jit(actor_apply_fn: Callable[..., distrax.Distribution],
                     actor_params: Params,
                     observations: np.ndarray,
                     actor_batch_stats: Any) -> jnp.ndarray:
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist = actor_apply_fn(input_collections, observations, training=False,
                          mutable=False)
    return dist.mode()

@partial(jax.jit, static_argnames=('actor_apply_fn', 'num_actions'))
def eval_actions_jit_cem(
        actor_apply_fn: Callable[..., distrax.Distribution], 
        actor_params: Params, 
        observations: np.ndarray, 
        actor_batch_stats: Any,
        critic_encoder,
        critic_decoder,
        rng: PRNGKey,
        num_actions=50,
    ) -> jnp.ndarray:
    print('USING CEM')
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats

    input_collections_ce = {'params': critic_encoder.params}
    if critic_encoder.batch_stats is not None:
        input_collections_ce['batch_stats'] = critic_encoder.batch_stats
    
    input_collections_cd = {'params': critic_decoder.params}
    if critic_decoder.batch_stats is not None:
        input_collections_cd['batch_stats'] = critic_decoder.batch_stats

    embed_obs = critic_encoder.apply_fn(input_collections_ce, observations, training=False)

    emb_pixels_tiled = jnp.tile(embed_obs['pixels'], (num_actions,1))
    emb_state_tiled = jnp.tile(embed_obs['state'], (num_actions, 1, 1))
    emb_task_id_tiled = jnp.tile(embed_obs['task_id'], (num_actions, 1))
    emb_obs_tiled = dict(pixels=emb_pixels_tiled, state=emb_state_tiled, task_id=emb_task_id_tiled)


    pixels_tiled = jnp.tile(observations['pixels'], (num_actions,1,1,1,1))
    state_tiled = jnp.tile(observations['state'], (num_actions, 1, 1))
    task_id_tiled = jnp.tile(observations['task_id'], (num_actions, 1))
    obs_tiled = dict(pixels=pixels_tiled, state=state_tiled, task_id=task_id_tiled)

    dist = actor_apply_fn(input_collections, obs_tiled, training=False, mutable=False)
    rng, key = jax.random.split(rng)
    actions = dist.sample(seed=key)

    q_values = critic_decoder.apply_fn(input_collections_cd, emb_obs_tiled, actions, training=False, mutable=False)
    q_values = q_values.min(0)
    
    des_shape = actor_apply_fn(input_collections, observations, training=False, mutable=False).mode().shape
    return jnp.reshape(actions[jnp.argmax(q_values)], des_shape)

@partial(jax.jit, static_argnames='actor_apply_fn')
def sample_actions_jit(
        rng: PRNGKey, actor_apply_fn: Callable[..., distrax.Distribution],
        actor_params: Params,
        observations: np.ndarray,
        actor_batch_stats: Any) -> Tuple[PRNGKey, jnp.ndarray]:
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist = actor_apply_fn(input_collections, observations)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)


@partial(jax.jit, static_argnames='actor_apply_fn')
def sample_actions_jit(
        rng: PRNGKey, actor_apply_fn: Callable[..., distrax.Distribution],
        actor_params: Params,
        observations: np.ndarray,
        actor_batch_stats: Any) -> Tuple[PRNGKey, jnp.ndarray]:
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist = actor_apply_fn(input_collections, observations)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)
