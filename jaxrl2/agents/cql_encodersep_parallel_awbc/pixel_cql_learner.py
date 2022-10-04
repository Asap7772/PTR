"""Implementations of algorithms for continuous control."""
from audioop import cross
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flax.training import checkpoints
import pathlib

import copy
import functools
from typing import Dict, Optional, Sequence, Tuple
import flax

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state

from jaxrl2.agents.agent import Agent

from jaxrl2.agents.cql_encodersep_parallel_awbc.actor_updater import update_actor
from jaxrl2.agents.cql_encodersep_parallel_awbc.critic_updater import update_critic, extend_and_repeat, reshape_for_cql_computation, get_cds_weights
from jaxrl2.agents.cql_encodersep_parallel_awbc.temperature_updater import update_temperature
from jaxrl2.agents.cql_encodersep_parallel_awbc.temperature import Temperature
from jaxrl2.agents.cql_encodersep_parallel_awbc.actor_updater_adv import log_prob_update
from jaxrl2.agents.cql_encodersep_parallel_awbc.actor_updater_bc import log_prob_update_bc

from jaxrl2.utils.target_update import soft_target_update
from jaxrl2.types import Params, PRNGKey

from jaxrl2.agents.agent import Agent
from jaxrl2.networks.learned_std_normal_policy import LearnedStdNormalPolicy, LearnedStdTanhNormalPolicy
from jaxrl2.agents.drq.augmentations import batched_random_crop, color_transform
from jaxrl2.agents.common import _unpack
from jaxrl2.agents.drq.drq_learner import _share_encoder
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer, PixelMultiplexerEncoder, PixelMultiplexerDecoder, AuxPixelMultiplexerDecoder
from jaxrl2.networks.encoders.networks import PixelMultiplexerDecoderWithDropout, PixelMultiplexerEncoderWithoutFinal
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder, BiggerImpalaEncoder, BiggestImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall, ResNet50
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.autoregressive_policy import ARPolicy
from jaxrl2.networks.normal_policy import NormalPolicy
from jaxrl2.networks.values import StateActionEnsemble, StateValue, AuxStateActionEnsemble, StateActionEnsembleAutoregressive
from jaxrl2.networks.values.state_action_value import StateActionValue, AuxStateActionValue
from jaxrl2.networks.values.state_value import StateValueEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update
from flax.core import frozen_dict
import wandb

NUM_DATA_SHARE = 2

import numpy as np
from typing import Any

class TrainState(train_state.TrainState):
    batch_stats: Any = None

def repeat_dict(tensor, axis, repeat):
    if isinstance(tensor, frozen_dict.FrozenDict):
        new_tensor = {}
        for key in tensor:
            new_tensor[key] = repeat_dict(tensor[key], axis, repeat)
        new_tensor = tensor.copy(add_or_replace=new_tensor)
        return new_tensor
    else:
        return jnp.repeat(tensor, repeat, axis=axis)

def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    return state.replace(batch_stats=jax.lax.pmean(state.batch_stats, axis_name='pmap'))

@functools.partial(jax.pmap, static_broadcasted_argnums=list(range(9,31)), axis_name='pmap')
def _update_jit(
    rng: PRNGKey, actor: TrainState, adv_actor: TrainState, critic_encoder: TrainState,
    critic_decoder: TrainState, target_critic_encoder_params: Params, 
    target_critic_decoder_params: Params, temp: TrainState, batch: TrainState,
    discount: float, tau: float, target_entropy: float, backup_entropy: bool,
    critic_reduction: str, cql_alpha: float, max_q_backup: bool,
    dr3_coefficient: float, color_jitter: bool, cross_norm:bool, aug_next:bool,
    basis_projection_coefficient: float, use_basis_projection: bool, use_gaussian_policy: bool, 
    autoregressive_policy:bool, min_q_version: int, use_uds: int, use_cds:int, reward_min:float, 
    adv_start: int, bc_phase: int, adv_temp:int):

    # Comment out when using the naive replay buffer
    # batch = _unpack(batch)
    
    if use_uds:
        orig_masks=batch['masks']
        # not sure if I should be only doing this for the critic update
        axis, repeat = 0, NUM_DATA_SHARE
        assert repeat > 1, "repeat must be greater than 1"     
        
        observations_tiled = repeat_dict(batch['observations'], axis=axis, repeat=repeat)
        next_observations_tiled = repeat_dict(batch['next_observations'], axis=axis, repeat=repeat)
        
        # change axis for remaining batch variables
        
        orig_task_ids = batch['observations']['task_id']
        num_task_ids = batch['observations']['task_id'].shape[-1] # should be the same for all tasks
        which_task_id = jnp.argmax(orig_task_ids, axis=-1) # taskids for histogram
        
        task_ids_other= jnp.repeat(batch['observations']['task_id'],repeat-1, axis=axis)
        rng, key = jax.random.split(rng, 2)
        sample_random_task_ids = jax.random.randint(key, task_ids_other.shape, 0, num_task_ids-1)
        sample_random_task_ids = jnp.where(sample_random_task_ids < orig_task_ids, sample_random_task_ids, sample_random_task_ids+1)
        task_ids = jnp.concatenate([orig_task_ids, task_ids_other], axis=axis)
        
        observations_tiled = observations_tiled.copy(add_or_replace={'task_id': task_ids})
        next_observations_tiled = next_observations_tiled.copy(add_or_replace={'task_id': task_ids})
        
        actions = jnp.repeat(batch['actions'],repeat, axis=axis)
        masks = jnp.repeat(batch['masks'],repeat, axis=axis)
        
        rewards_other= jnp.repeat(jnp.zeros_like(batch['rewards']),repeat-1, axis=axis)
        rewards_other = rewards_other + reward_min # set to minimum reward
        rewards = jnp.concatenate([batch['rewards'], rewards_other], axis=axis)
        
        batch = batch.copy(add_or_replace=dict(
            observations=observations_tiled,
            next_observations=next_observations_tiled,
            actions=actions,
            masks=masks,
            rewards=rewards,
        ))
    
    if use_cds:
        # relabel_masks is now 0 for the correct task, 1 for the wrong task
        relabel_masks = batch['masks']
        cds_weight, cds_weight1, cds_weight2, cds_q1_entropy, cds_q2_entropy = get_cds_weights(key, actor,critic_encoder, critic_decoder)
    else:
        cds_weight = cds_weight1 = cds_weight2 = 1.0
        cds_q1_entropy, cds_q2_entropy = 0.0, 0.0
        
    # shuffle batch?
    # batch_size = batch['actions'].shape[0]
    # rng, key = jax.random.split(rng, 2)
    # shuffled_indices = jnp.permutation(key, batch_size)
    
    rng, key = jax.random.split(rng)
    aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

    if color_jitter:
        rng, key = jax.random.split(rng)
        aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
    observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
    batch = batch.copy(add_or_replace={'observations': observations})

    print ('Aug next or not: ', aug_next)
    if aug_next:
        rng, key = jax.random.split(rng)
        aug_next_pixels = batched_random_crop(key, batch['next_observations']['pixels'])
        if color_jitter:
            rng, key = jax.random.split(rng)
            aug_next_pixels = (color_transform(key, aug_next_pixels.astype(jnp.float32) / 255.) * 255).astype(jnp.uint8)
        next_observations = batch['next_observations'].copy(
            add_or_replace={'pixels': aug_next_pixels})
        batch = batch.copy(add_or_replace={'next_observations': next_observations})

    if adv_start:
        new_critic_decoder, new_critic_encoder, new_actor, new_temp, new_target_critic_encoder_params, new_target_critic_decoder_params = \
            critic_decoder, critic_encoder, actor, temp, target_critic_encoder_params, target_critic_decoder_params
            
        actor_info = critic_info = alpha_info = {}
        
        new_adv_actor, adv_actor_info = log_prob_update(key, actor, new_critic_encoder, new_critic_decoder, batch, temperature=adv_temp)
    else:
        key, rng = jax.random.split(rng)
        target_critic_encoder = critic_encoder.replace(params=target_critic_encoder_params)
        target_critic_decoder = critic_decoder.replace(params=target_critic_decoder_params)
        
        (new_critic_encoder, new_critic_decoder), critic_info = update_critic(
            key,
            actor,
            critic_encoder,
            critic_decoder,
            target_critic_encoder,
            target_critic_decoder,
            temp,
            batch,
            discount,
            backup_entropy=backup_entropy,
            critic_reduction=critic_reduction,
            cql_alpha=cql_alpha,
            max_q_backup=max_q_backup,
            dr3_coefficient=dr3_coefficient,
            cross_norm=cross_norm,
            use_basis_projection=use_basis_projection,
            basis_projection_coefficient=basis_projection_coefficient,
            use_gaussian_policy=use_gaussian_policy,
            min_q_version=min_q_version
        )
        if hasattr(new_critic_encoder, 'batch_stats') and new_critic_encoder.batch_stats is not None:
            print ('Syncing batch stats for critic encoder')
            new_critic_encoder = sync_batch_stats(new_critic_encoder)
        if hasattr(new_critic_decoder, 'batch_stats') and new_critic_decoder.batch_stats is not None:
            print ('Syncing batch stats for critic decoder')
            new_critic_decoder = sync_batch_stats(new_critic_decoder)
        
        new_target_critic_encoder_params = soft_target_update(new_critic_encoder.params, target_critic_encoder_params, tau)
        new_target_critic_decoder_params = soft_target_update(new_critic_decoder.params, target_critic_decoder_params, tau)

        rng, key = jax.random.split(rng)
        new_actor, actor_info = update_actor(key, actor, new_critic_encoder, new_critic_decoder, temp, batch, cross_norm=cross_norm,
                                            use_gaussian_policy=use_gaussian_policy, autoregressive_policy=autoregressive_policy)
        
        if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
            actor = sync_batch_stats(actor)
        
        new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)
        
        # mse bc initialization for awbc actor
        if bc_phase:
            new_adv_actor, adv_actor_info = log_prob_update_bc(key, new_adv_actor, new_critic_encoder, new_critic_decoder, batch, temperature=adv_temp)
        else:
            new_adv_actor, adv_actor_info = adv_actor, {}
        
    if use_uds:
        task_id_dict={
            'task_ids': which_task_id,
            'num_tasks': num_task_ids,
        }
    else:
        task_id_dict={}

    info_dict = {
        **critic_info,
        **actor_info,
        **adv_actor_info,
        **alpha_info,
        **task_id_dict,
    }

    return rng, new_actor, new_adv_actor, (new_critic_encoder, new_critic_decoder), (new_target_critic_encoder_params, new_target_critic_decoder_params), new_temp, info_dict


class PixelAWBCLearnerEncoderSepParallel(Agent):
    
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 100,
                 discount: float = 0.99,
                 cql_alpha: float = 0.0,
                 tau: float = 0.0,
                 backup_entropy: bool = False,
                 target_entropy: Optional[float] = None,
                 critic_reduction: str = 'min',
                 dropout_rate: Optional[float] = None,
                 init_temperature: float = 1.0,
                 max_q_backup: bool = False,
                 policy_encoder_type: str = 'resnet_small',
                 encoder_type: str ='resnet_small',
                 encoder_norm: str = 'batch',
                 dr3_coefficient: float = 0.0,
                 method:bool = False,
                 method_const:float = 0.0,
                 method_type:int=0,
                 cross_norm:bool = False,
                 use_spatial_softmax=False,
                 softmax_temperature=-1,
                 use_spatial_learned_embeddings=False,
                 share_encoders=False,
                 color_jitter=True,
                 use_bottleneck=True,
                 aug_next=True,
                 use_action_sep=False,
                 use_basis_projection=False,
                 basis_projection_coefficient=0.0,
                 use_multiplicative_cond=False,
                 policy_use_multiplicative_cond=False,
                 target_entropy_factor=1,
                 use_gaussian_policy=False,
                 use_normalized_features=False,
                 use_pixel_sep=False,
                 min_q_version=3,
                 std_scale_for_gaussian_policy=0.05,
                 q_dropout_rate=0.0,
                 use_uds=0, 
                 use_cds=0,
                 min_reward=-1,
                 bc_policy_end=80000,
                 adv_policy_start=0,
                 adv_temp=1.0,
                 autoregressive_policy=0,
                 autoregressive_repeat=0,
                 autoregressive_project=0,
                 autoregressive_qfunc=0,
                 autoregressive_type=0,
        ):
        self.min_reward=min_reward
        self.color_jitter=color_jitter
        self.autoregressive_policy = autoregressive_policy
        
        self.bc_policy_end = bc_policy_end
        self.adv_policy_start = adv_policy_start
        self.adv_temp = adv_temp
        
        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim * target_entropy_factor
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy
        self.critic_reduction = critic_reduction

        self.tau = tau
        self.discount = discount
        self.max_q_backup = max_q_backup
        self.dr3_coefficient = dr3_coefficient

        self.method = method
        self.method_const = method_const
        self.method_type = method_type
        self.cross_norm = cross_norm
        self.is_replicated=True
        self.use_action_sep = use_action_sep
        self.use_basis_projection = use_basis_projection
        self.basis_projection_coefficient = basis_projection_coefficient
        self.use_multiplicative_cond = use_multiplicative_cond
        self.use_spatial_learned_embeddings = use_spatial_learned_embeddings
        self.use_gaussian_policy = use_gaussian_policy
        self.use_normalized_features = use_normalized_features
        self.policy_use_multiplicative_cond = policy_use_multiplicative_cond
        self.use_pixel_sep = use_pixel_sep
        self.min_q_version = min_q_version
        self.q_dropout_rate = q_dropout_rate
        self.use_uds, self.use_cds = use_uds, use_cds
        
        assert not (use_uds and use_cds), 'Cannot use both UDS and CDS'
        print('DATA SHARING', use_uds, use_cds)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, adv_actor_key, critic_key, temp_key = jax.random.split(rng, 5)

        if encoder_type == 'small':
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == 'impala':
            print('using impala')
            encoder_def = ImpalaEncoder(use_multiplicative_cond=use_multiplicative_cond)
        elif encoder_type == 'impala_bigger':
            print('using bigger impala')
            encoder_def = BiggerImpalaEncoder(use_multiplicative_cond=use_multiplicative_cond)
        elif encoder_type == 'impala_biggest':
            print('using biggest impala')
            encoder_def = BiggestImpalaEncoder(use_multiplicative_cond=use_multiplicative_cond)
        elif encoder_type == 'resnet_small':
            encoder_def = ResNetSmall(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature,
                                      use_spatial_learned_embeddings=use_spatial_learned_embeddings)
        elif encoder_type == 'resnet_18_v1':
            encoder_def = ResNet18(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature,
                                   use_multiplicative_cond=use_multiplicative_cond,
                                   use_spatial_learned_embeddings=use_spatial_learned_embeddings,
                                   num_spatial_blocks=8)
        elif encoder_type == 'resnet_34_v1':
            encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature,
                                   use_multiplicative_cond=use_multiplicative_cond,
                                   use_spatial_learned_embeddings=use_spatial_learned_embeddings,
                                   num_spatial_blocks=8)
        elif encoder_type == 'resnet_50_v1':
            encoder_def = ResNet50(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature,
                                   use_multiplicative_cond=use_multiplicative_cond,
                                   use_spatial_learned_embeddings=use_spatial_learned_embeddings,
                                   num_spatial_blocks=4)
        elif encoder_type == 'resnet_small_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
        elif encoder_type == 'resnet_18_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif encoder_type == 'resnet_34_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        else:
            raise ValueError('encoder type not found!')
        
        if policy_encoder_type == 'small':
            policy_encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif policy_encoder_type == 'impala':
            print('using impala')
            policy_encoder_def = ImpalaEncoder(use_multiplicative_cond=policy_use_multiplicative_cond)
        elif policy_encoder_type == 'impala_bigger':
            print('using bigger impala')
            policy_encoder_def = BiggerImpalaEncoder(use_multiplicative_cond=policy_use_multiplicative_cond)
        elif policy_encoder_type == 'impala_biggest':
            print('using biggest impala')
            policy_encoder_def = BiggestImpalaEncoder(use_multiplicative_cond=policy_use_multiplicative_cond)
        elif policy_encoder_type == 'resnet_small':
            policy_encoder_def = ResNetSmall(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax,
                                             softmax_temperature=softmax_temperature,
                                             use_spatial_learned_embeddings=use_spatial_learned_embeddings)
        elif policy_encoder_type == 'resnet_18_v1':
            policy_encoder_def = ResNet18(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax,
                                          softmax_temperature=softmax_temperature,
                                          use_spatial_learned_embeddings=use_spatial_learned_embeddings,
                                          use_multiplicative_cond=policy_use_multiplicative_cond,
                                          num_spatial_blocks=8) 
        elif policy_encoder_type == 'resnet_34_v1':
            policy_encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax,
                                          softmax_temperature=softmax_temperature,
                                          use_spatial_learned_embeddings=use_spatial_learned_embeddings,
                                          use_multiplicative_cond=policy_use_multiplicative_cond,
                                          num_spatial_blocks=8)
        elif policy_encoder_type == 'resnet_50_v1':
            policy_encoder_def = ResNet50(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax,
                                          softmax_temperature=softmax_temperature,
                                          use_spatial_learned_embeddings=use_spatial_learned_embeddings,
                                          use_multiplicative_cond=policy_use_multiplicative_cond,
                                          num_spatial_blocks=4)
        elif policy_encoder_type == 'resnet_small_v2':
            policy_encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
        elif policy_encoder_type == 'resnet_18_v2':
            policy_encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif policy_encoder_type == 'resnet_34_v2':
            policy_encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        elif policy_encoder_type == 'same':
            policy_encoder_def = encoder_def
            policy_use_multiplicative_cond = use_multiplicative_cond
        else:
            raise ValueError('encoder type not found!')
        
        adv_policy_encoder_def = policy_encoder_def

        self.std_scale_for_gaussian_policy = std_scale_for_gaussian_policy
        if autoregressive_policy:
            print('using autoregressive policy')
            policy_def = ARPolicy(hidden_dims, 
                                  action_dim, 
                                  dropout_rate=dropout_rate,
                                  repeat=autoregressive_repeat,
                                  project=autoregressive_project)
        elif self.use_gaussian_policy:
            policy_def = NormalPolicy(hidden_dims, action_dim, dropout_rate=dropout_rate,
                                      std=self.std_scale_for_gaussian_policy)
        else:
            policy_def = LearnedStdTanhNormalPolicy(hidden_dims,action_dim, dropout_rate=dropout_rate)
        
        adv_policy_def = LearnedStdNormalPolicy(hidden_dims, action_dim, dropout_rate=dropout_rate) # learned std gaussian policy

        actor_def = PixelMultiplexer(encoder=policy_encoder_def,
                                     network=policy_def,
                                     latent_dim=latent_dim,
                                     stop_gradient=share_encoders,
                                     use_bottleneck=use_bottleneck,
                                     use_multiplicative_cond=policy_use_multiplicative_cond)
        
        adv_actor_def = PixelMultiplexer(encoder=adv_policy_encoder_def,
                                         network=adv_policy_def,
                                         latent_dim=latent_dim,
                                         stop_gradient=share_encoders,
                                         use_bottleneck=use_bottleneck,
                                         use_multiplicative_cond=policy_use_multiplicative_cond)

        actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init['params']
        actor_batch_stats = actor_def_init['batch_stats'] if 'batch_stats' in actor_def_init else None
        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  batch_stats=actor_batch_stats,
                                  tx=optax.adam(learning_rate=actor_lr))
        
        
        adv_actor_def_init = actor_def.init(adv_actor_key, observations)
        adv_actor_params = adv_actor_def_init['params']
        adv_actor_batch_stats = adv_actor_def_init['batch_stats'] if 'batch_stats' in adv_actor_def_init else None
        adv_actor = TrainState.create(apply_fn=adv_actor_def.apply,
                                  params=adv_actor_params,
                                  batch_stats=adv_actor_batch_stats,
                                  tx=optax.adam(learning_rate=actor_lr))
        
        actor = flax.jax_utils.replicate(actor)
        adv_actor = flax.jax_utils.replicate(adv_actor)

        if self.use_basis_projection:
            network_def = AuxStateActionEnsemble(hidden_dims, num_qs=2, use_action_sep=self.use_action_sep,
                                                 use_pixel_sep=self.use_pixel_sep)
        elif autoregressive_qfunc:
            network_def = StateActionEnsembleAutoregressive(hidden_dims, num_qs=2, use_action_sep=self.use_action_sep,
                                              use_normalized_features=use_normalized_features,
                                              use_pixel_sep=self.use_pixel_sep, autoregressive_type=autoregressive_type)
        else:
            network_def = StateActionEnsemble(hidden_dims, num_qs=2, use_action_sep=self.use_action_sep,
                                              use_normalized_features=use_normalized_features,
                                              use_pixel_sep=self.use_pixel_sep)

        if self.q_dropout_rate > 0.0:
            critic_def_encoder = PixelMultiplexerEncoderWithoutFinal(
                encoder=encoder_def, latent_dim=latent_dim,
                use_bottleneck=use_bottleneck,
                use_multiplicative_cond=use_multiplicative_cond
            )
        else:
            critic_def_encoder = PixelMultiplexerEncoder(
                encoder=encoder_def,latent_dim=latent_dim, use_bottleneck=use_bottleneck,
                use_multiplicative_cond=use_multiplicative_cond)
        
        if self.use_basis_projection:
            critic_def_decoder = AuxPixelMultiplexerDecoder(network=network_def)
        elif self.q_dropout_rate > 0.0:
            critic_def_decoder = PixelMultiplexerDecoderWithDropout(
                network=network_def, 
                dropout_rate=self.q_dropout_rate
            )
        else:
            critic_def_decoder = PixelMultiplexerDecoder(network=network_def)
        
        critic_key_encoder, critic_key_decoder = jax.random.split(critic_key, 2)
        critic_def_encoder_init = critic_def_encoder.init(critic_key_encoder, observations)
        critic_encoder_params = critic_def_encoder_init['params']
        critic_encoder_batch_stats = critic_def_encoder_init['batch_stats'] if 'batch_stats' in critic_def_encoder_init else None
        
        if 'batch_stats' in critic_def_encoder_init:
            embed_obs, _ = critic_def_encoder.apply(
                {'params': critic_encoder_params,
                'batch_stats': critic_def_encoder_init['batch_stats']}, observations, mutable=['batch_stats'])
        else:
            embed_obs = critic_def_encoder.apply({'params': critic_encoder_params}, observations)
        
        critic_def_decoder_init = critic_def_decoder.init(critic_key_decoder, embed_obs, actions)
        critic_decoder_params = critic_def_decoder_init['params']
        critic_decoder_batch_stats = critic_def_decoder_init['batch_stats'] if 'batch_stats' in critic_def_decoder_init else None

        critic_encoder = TrainState.create(apply_fn=critic_def_encoder.apply,
                                   params=critic_encoder_params,
                                   batch_stats=critic_encoder_batch_stats,
                                   tx=optax.adam(learning_rate=critic_lr))
        
        critic_decoder = TrainState.create(apply_fn=critic_def_decoder.apply,
                                   params=critic_decoder_params,
                                   batch_stats=critic_decoder_batch_stats,
                                   tx=optax.adam(learning_rate=critic_lr))
        
        critic_encoder = flax.jax_utils.replicate(critic_encoder)
        critic_decoder = flax.jax_utils.replicate(critic_decoder)

        target_critic_encoder_params = copy.deepcopy(critic_encoder.params)
        target_critic_decoder_params = copy.deepcopy(critic_decoder.params)

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                                 params=temp_params,
                                 tx=optax.adam(learning_rate=temp_lr))
        temp = flax.jax_utils.replicate(temp)

        self._rng = rng
        self._actor = actor
        self._adv_actor = adv_actor
        self._critic_encoder = critic_encoder
        self._critic_decoder = critic_decoder
        self._critic = (critic_encoder, critic_decoder)
        
        self.aug_next = aug_next
        
        self._temp = temp
        self._target_critic_encoder_params = target_critic_encoder_params
        self._target_critic_decoder_params = target_critic_decoder_params
        self._target_critic_params = (target_critic_encoder_params, target_critic_decoder_params)
        
        self._cql_alpha = cql_alpha
        self._curr_step = 0
        print ('Discount: ', self.discount)
        print ('CQL Alpha: ', self._cql_alpha)
        print('Method: ', self.method, 'Const: ', self.method_const)
        
    def unreplicate(self):
        if not self.is_replicated:
            raise RuntimeError('Not Replicated') 
        # else:
        #     print('UNREPLICATING '*5)
        self._actor = flax.jax_utils.unreplicate(self._actor)
        self._adv_actor = flax.jax_utils.unreplicate(self._adv_actor)
        self._critic_encoder = flax.jax_utils.unreplicate(self._critic_encoder)
        self._critic_decoder = flax.jax_utils.unreplicate(self._critic_decoder)
        self._target_critic_encoder_params = flax.jax_utils.unreplicate(self._target_critic_encoder_params)
        self._target_critic_decoder_params = flax.jax_utils.unreplicate(self._target_critic_decoder_params)
        self._critic = (self._critic_encoder, self._critic_decoder)
        self._target_critic_params = (self._target_critic_encoder_params, self._target_critic_decoder_params)
        self.is_replicated=False
    
    def replicate(self):
        if self.is_replicated:
            raise RuntimeError('Already Replicated') 
        # else:
        #     print('REPLICATING '*5)
        self._actor = flax.jax_utils.replicate(self._actor)
        self._adv_actor = flax.jax_utils.replicate(self._adv_actor)
        self._critic_encoder = flax.jax_utils.replicate(self._critic_encoder)
        self._critic_decoder = flax.jax_utils.replicate(self._critic_decoder)
        self._target_critic_encoder_params = flax.jax_utils.replicate(self._target_critic_encoder_params)
        self._target_critic_decoder_params = flax.jax_utils.replicate(self._target_critic_decoder_params)
        self._critic = (self._critic_encoder, self._critic_decoder)
        self._target_critic_params = (self._target_critic_encoder_params, self._target_critic_decoder_params)
        self.is_replicated=True


    def update(self, batch: FrozenDict):
        num_devices = len(jax.devices())
        new_rng, new_actor, new_adv_actor, new_critic, new_target_critic_params, new_temp, info = _update_jit(
            jax.random.split(self._rng, num_devices), self._actor, self._adv_actor, self._critic_encoder, self._critic_decoder, 
            self._target_critic_encoder_params, self._target_critic_decoder_params,
            self._temp, batch, self.discount, self.tau, self.target_entropy,
            self.backup_entropy, self.critic_reduction, self._cql_alpha, self.max_q_backup,
            self.dr3_coefficient, self.color_jitter, self.cross_norm, self.aug_next,
            self.basis_projection_coefficient, self.use_basis_projection, self.use_gaussian_policy, 
            self.autoregressive_policy, self.min_q_version, self.use_uds, self.use_cds, self.min_reward, 
            int(self._curr_step > self.adv_policy_start), int(self._curr_step < self.bc_policy_end), self.adv_temp)
        
        new_critic_encoder, new_critic_decoder = new_critic
        new_target_critic_encoder_params, new_target_critic_decoder_params = new_target_critic_params
        
        info = {k:v[0] for k,v in info.items()}

        self._rng = new_rng[0]
        self._actor = new_actor
        self._adv_actor = new_adv_actor
        self._critic_encoder = new_critic_encoder
        self._critic_decoder = new_critic_decoder
        self._critic = (new_critic_encoder, new_critic_decoder)
        
        self._target_critic_encoder_params = new_target_critic_encoder_params
        self._target_critic_decoder_params = new_target_critic_decoder_params
        self._target_critic_params = (new_target_critic_encoder_params, new_target_critic_decoder_params)
        
        self._temp = new_temp
        self._curr_step += 1
        
        return info

    def perform_eval(self, variant, i, wandb_logger, eval_buffer, eval_buffer_iterator, eval_env):
        # try:
        from examples.train_utils import make_multiple_value_reward_visulizations
        make_multiple_value_reward_visulizations(self, variant, i, eval_buffer, wandb_logger)
        # except Exception as e:
        #     print(e)
        #     print('Could not visualize')

    def show_value_reward_visulization(self, traj):
        q_vals = []
        for step in traj:
            q_vals.append(get_q_value(step['action'], step['observation'], self._critic_encoder, self._critic_decoder))
        rewards = [step['reward'] for step in traj]
        images = np.stack([step['observation']['pixels'][0] for step in traj], 0)
        make_visual_eval(q_vals, rewards, images, show_window=True)

    def make_value_reward_visulization(self, variant, trajs, **kwargs):
        # try:
        num_traj = len(trajs['rewards'])
        traj_images = []
        num_stack = variant.frame_stack

        for itraj in range(num_traj):
            observations = trajs['observations'][itraj]
            next_observations = trajs['next_observations'][itraj]
            actions = trajs['actions'][itraj]
            rewards = trajs['rewards'][itraj]
            masks = trajs['masks'][itraj]

            target_critic_encoder = self._critic_encoder.replace(params=self._target_critic_encoder_params)
            target_critic_decoder = self._critic_decoder.replace(params=self._target_critic_decoder_params)

            q_pred = []
            q_pis = []
            advantages = []
            target_q_pred = []
            bellman_loss = []
            task_ids = []
            
            traj_images = []
            
            # Do the frame stacking thing for observations
            # images = np.lib.stride_tricks.sliding_window_view(observations.pop('pixels'), num_stack + 1, axis=0)

            for t in range(0, len(actions)):
                action = actions[t][None]
                obs_pixels = observations['pixels'][t]
                next_obs_pixels = next_observations['pixels'][t]

                obs_dict = {'pixels': obs_pixels[None]}
                for k, v in observations.items():
                    if 'pixels' not in k:
                        obs_dict[k] = v[t][None]
                    
                next_obs_dict = {'pixels': next_obs_pixels[None]}
                for k, v in next_observations.items():
                    if 'pixels' not in k:
                        next_obs_dict[k] = v[t][None]
                
                q_value = get_q_value(action, obs_dict, self._critic_encoder, self._critic_decoder)
                curr_pi_action = get_action(obs_dict, self._actor)
                next_action = get_action(next_obs_dict, self._actor)
                target_q_value = get_q_value(next_action, next_obs_dict, target_critic_encoder, target_critic_decoder)
                target_q_value = rewards[t] + target_q_value.min() * self.discount * masks[t]
                
                q_pi = get_q_value(curr_pi_action, obs_dict, self._critic_encoder, self._critic_decoder)
                advantage = q_value - q_pi
                
                q_pis.append(q_pi)
                advantages.append(advantage)
                q_pred.append(q_value)
                target_q_pred.append(target_q_value.item())
                bellman_loss.append(((q_value-target_q_value)**2).mean().item())
                if 'task_id' in observations:
                    task_ids.append(np.argmax(observations['task_id']))
            
            # print ('lengths for verification: ', len(task_ids), len(q_pred), len(masks), len(bellman_loss))

            traj_images.append(make_visual(q_pred, rewards, observations['pixels'], masks, target_q_pred, bellman_loss, task_ids, q_pis, advantages))
        print('finished reward value visuals.')
        return np.concatenate(traj_images, 0)
        # except Exception as e:
        #     print(e)
        #     return np.zeros((num_traj, 128, 128, 3))

    @property
    def _save_dict(self):
        save_dict = {
            'critic': self._critic,
            'target_critic_params': self._target_critic_params,
            'actor': self._actor,
            'adv_actor': self._adv_actor,
            'temp': self._temp
        }
        return save_dict

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).is_file(), 'path {} not found!'.format(dir)
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)

        self._actor = output_dict['actor']
        self._adv_actor = output_dict['adv_actor']
        self._critic_encoder, self._critic_decoder = output_dict['critic']
        self._critic=(self._critic_encoder, self._critic_decoder)
        self._target_critic_encoder_params, self._target_critic_decoder_params = output_dict['target_critic_params']
        self._target_critic_params = (self._target_critic_encoder_params, self._target_critic_decoder_params)
        self._temp = output_dict['temp']
        
        # note that temp is still replicated
        
        self.is_replicated = False # stored in the checkpoint, so we need to reset this
        
        print('restored from ', dir)
        

@functools.partial(jax.jit)
def get_action(obs_dict, actor):
    # print(f'{images.shape=}')
    # print(f'{images[..., None]=}')
    key_dropout, key_pi = jax.random.split(jax.random.PRNGKey(0))
    
    if actor.batch_stats is not None:
        dist = actor.apply_fn({'params': actor.params, 'batch_stats': actor.batch_stats}, obs_dict, rngs={'dropout': key_dropout})
    else:
        dist = actor.apply_fn({'params': actor.params}, obs_dict, rngs={'dropout': key_dropout})
        
    actions, policy_log_probs = dist.sample_and_log_prob(seed=key_pi)
    return actions

@functools.partial(jax.jit)
def get_q_value(actions, obs_dict, critic_encoder, critic_decoder):
    if critic_encoder.batch_stats is not None:
        embed_obs, _ = critic_encoder.apply_fn({'params': critic_encoder.params, 'batch_stats': critic_encoder.batch_stats}, obs_dict, mutable=['batch_stats'])
    else:    
        embed_obs = critic_encoder.apply_fn({'params': critic_encoder.params}, obs_dict)
        
    if critic_decoder.batch_stats is not None:
        q_pred, _ = critic_decoder.apply_fn({'params': critic_decoder.params, 'batch_stats': critic_decoder.batch_stats}, embed_obs, actions, mutable=['batch_stats'])
    else:    
        q_pred = critic_decoder.apply_fn({'params': critic_decoder.params}, embed_obs, actions)
        
    return q_pred

def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr

def make_visual(q_estimates, rewards, images, masks, target_q_pred, bellman_loss, task_ids, q_pis, advantages):
    q_estimates_np = np.stack(q_estimates, 0)
    q_pis_np = np.stack(q_pis, 0)
    advantages_np = np.stack(advantages, 0)

    # abridged images
    fig, axs = plt.subplots(10, 1, figsize=(8, 15))
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(q_estimates)])

    # assume image in T, C, H, W shape
    assert len(images.shape) == 5
    images = images[..., -1]  # only taking the most recent image of the stack
    assert images.shape[-1] == 3

    interval = images.shape[0] // 4
    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)
    
    axs[0].imshow(sel_images)
    axs[1].imshow(np.concatenate(np_unstack(images, 0), 1)) # all images
    
    axs[2].plot(q_estimates_np[:, 0], linestyle='--', marker='o')
    axs[2].plot(q_estimates_np[:, 1], linestyle='--', marker='o')
    axs[2].set_ylabel('q values')
    
    axs[3].plot(q_pis_np[:, 0], linestyle='--', marker='o')
    axs[3].plot(q_pis_np[:, 1], linestyle='--', marker='o')
    axs[3].set_ylabel('q pi values')
    
    axs[4].plot(advantages_np[:, 0], linestyle='--', marker='o')
    axs[4].plot(advantages_np[:, 1], linestyle='--', marker='o')
    axs[4].set_ylabel('advantages values')
    
    axs[5].plot(target_q_pred, linestyle='--', marker='o')
    axs[5].set_ylabel('target_q_pred')
    axs[5].set_xlim([0, len(target_q_pred)])
    
    axs[6].plot(bellman_loss, linestyle='--', marker='o')
    axs[6].set_ylabel('bellman_loss')
    axs[6].set_xlim([0, len(bellman_loss)])
    
    axs[7].plot(rewards, linestyle='--', marker='o')
    axs[7].set_ylabel('rewards')
    axs[7].set_xlim([0, len(rewards)])
    
    axs[8].plot(masks, linestyle='--', marker='o')
    axs[8].set_ylabel('masks')
    axs[8].set_xlim([0, len(masks)])
    
    if len(task_ids) > 0:
        axs[9].plot(task_ids, linestyle='--', marker='o')
        axs[9].set_ylabel('task_ids')
        axs[9].set_xlim([0, len(masks)])
    else:
        axs[9].plot(masks, linestyle='--', marker='o')
        axs[9].set_ylabel('masks')
        axs[9].set_xlim([0, len(masks)])

    plt.tight_layout()

    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return out_image


def make_visual_eval(q_estimates, rewards, images, masks=None, show_window=False):
    if show_window:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
    else:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

    q_estimates_np = np.stack(q_estimates, 0)

    fig, axs = plt.subplots(4, 1)
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(q_estimates)])

    # assume image in T, H, W, C, 1 shape
    assert len(images.shape) == 5
    images = images[..., -1]  # only taking the most recent image of the stack
    assert images.shape[-1] == 3

    interval = images.shape[0] // 4
    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)

    axs[0].imshow(sel_images)
    axs[1].plot(q_estimates_np[:, 0], linestyle='--', marker='o')
    axs[1].plot(q_estimates_np[:, 1], linestyle='--', marker='o')
    axs[1].set_ylabel('q values')
    axs[2].plot(rewards, linestyle='--', marker='o')
    axs[2].set_ylabel('rewards')
    axs[2].set_xlim([0, len(rewards)])
    if masks is not None:
        axs[3].plot(masks, linestyle='--', marker='o')
        axs[3].set_ylabel('masks')
        axs[3].set_xlim([0, len(masks)])
    else:
        masks = 1-np.array(rewards)
        masks = masks.tolist()
        axs[3].plot(masks, linestyle='--', marker='o')
        axs[3].set_ylabel('masks')
        axs[3].set_xlim([0, len(masks)])

    plt.tight_layout()

    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if show_window:
        plt.imshow(out_image)
        plt.show()
    plt.close(fig)
    return out_image
