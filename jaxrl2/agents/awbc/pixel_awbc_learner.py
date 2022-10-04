"""Implementations of algorithms for continuous control."""
import matplotlib
matplotlib.use('Agg')
from flax.training import checkpoints
import pathlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
import copy
import functools
from typing import Dict, Optional, Sequence, Tuple, Union
from jaxrl2.data.dataset import DatasetDict

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from typing import Any
import json
from jaxrl2.networks.learned_std_normal_policy import LearnedStdNormalPolicy, LearnedStdTanhNormalPolicy

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.drq.augmentations import batched_random_crop, color_transform
from jaxrl2.agents.common import _unpack
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.agents.awbc.actor_updater import log_prob_update
from jaxrl2.networks.normal_policy import NormalPolicy
from jaxrl2.types import Params, PRNGKey
from jaxrl2.networks.values import StateActionEnsemble, StateValue
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer, PixelMultiplexerEncoder, PixelMultiplexerDecoder

class TrainState(train_state.TrainState):
    batch_stats: Any

@functools.partial(jax.jit, static_argnames=('color_jitter'))
def _update_jit(
    rng: PRNGKey, actor: TrainState, critic_encoder: TrainState, critic_decoder: TrainState, batch: TrainState, color_jitter: bool
) -> Tuple[PRNGKey, TrainState, Dict[str, float]]:
    # batch = _unpack(batch)

    rng, key = jax.random.split(rng)
    aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

    if color_jitter:
        rng, key = jax.random.split(rng)
        aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
    observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
    batch = batch.copy(add_or_replace={'observations': observations})

    aug_next=False
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

    rng, key = jax.random.split(rng)
    _, new_actor, actor_info = log_prob_update(key, actor, critic_encoder, critic_decoder, batch)

    return rng, new_actor,{**actor_info, 'pixels': aug_pixels}

class PixelAWBCLearner(Agent):

    def __init__(self,
                 seed: int,
                 observations: Union[jnp.ndarray, DatasetDict],
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 dropout_rate: Optional[float] = None,
                 policy_encoder_type: str = 'resnet_small',
                 encoder_type: str ='resnet_small',
                 encoder_norm: str = 'batch',
                 use_spatial_softmax=True,
                 softmax_temperature=1.0,
                 use_bottleneck=True,
                 color_jitter = True,
                 use_multiplicative_cond=False,
                 use_spatial_learned_embeddings=False,
                 use_normalized_features=False,
                 use_pixel_sep=True,
                 use_action_sep=False,
                 dir=None,
                 use_gaussian_policy=False,
                 ):
        self.color_jitter = color_jitter
        self.use_spatial_learned_embeddings = use_spatial_learned_embeddings
        print('use_spatial_learned_embeddings: ', use_spatial_learned_embeddings)

        action_dim = actions.shape[-1]
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)
        
        if dir is not None:
            path = '/'.join(dir.split('/')[:-1])
            config_file = path + '/config.json'
            with open(config_file) as f:
                config = json.load(f)
        else:
            config = {}

        if encoder_type == 'small':
            encoder_def = Encoder(config.get('cnn_features', cnn_features), config.get('cnn_strides', cnn_strides), config.get('cnn_padding', cnn_padding))
        elif encoder_type == 'impala':
            print('using impala')
            encoder_def = ImpalaEncoder()
        elif encoder_type == 'resnet_small':
            encoder_def = ResNetSmall(norm=config.get('encoder_norm', encoder_norm), use_spatial_softmax=config.get('use_spatial_softmax', use_spatial_softmax), softmax_temperature=config.get('softmax_temperature', softmax_temperature))
        elif encoder_type == 'resnet_18_v1':
            encoder_def = ResNet18(norm=config.get('encoder_norm', encoder_norm), use_spatial_softmax=config.get('use_spatial_softmax', use_spatial_softmax), softmax_temperature=config.get('softmax_temperature', softmax_temperature))
        elif encoder_type == 'resnet_34_v1':
            encoder_def = ResNet34(norm=config.get('encoder_norm', encoder_norm), use_spatial_softmax=config.get('use_spatial_softmax', use_spatial_softmax), softmax_temperature=config.get('softmax_temperature', softmax_temperature))
        elif encoder_type == 'resnet_small_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=config.get('encoder_norm', encoder_norm))
        elif encoder_type == 'resnet_18_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=config.get('encoder_norm', encoder_norm))
        elif encoder_type == 'resnet_34_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=config.get('encoder_norm', encoder_norm))
        else:
            raise ValueError('encoder type not found!')
        
        if policy_encoder_type == 'small':
            policy_encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif policy_encoder_type == 'impala':
            print('using impala')
            policy_encoder_def = ImpalaEncoder()
        elif policy_encoder_type == 'resnet_small':
            policy_encoder_def = ResNetSmall(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif policy_encoder_type == 'resnet_18_v1':
            policy_encoder_def = ResNet18(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif policy_encoder_type == 'resnet_34_v1':
            policy_encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif policy_encoder_type == 'resnet_small_v2':
            policy_encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
        elif policy_encoder_type == 'resnet_18_v2':
            policy_encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif policy_encoder_type == 'resnet_34_v2':
            policy_encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        elif policy_encoder_type == 'same':
            policy_encoder_def = encoder_def
        else:
            raise ValueError('encoder type not found!')

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)
        
        if use_gaussian_policy:
            policy_def = NormalPolicy(hidden_dims,
                                    action_dim,
                                    dropout_rate=dropout_rate)
        else:
            policy_def = LearnedStdTanhNormalPolicy(hidden_dims,
                                                    action_dim, 
                                                    dropout_rate=dropout_rate)

        actor_def = PixelMultiplexer(encoder=policy_encoder_def,
                                     network=policy_def,
                                     latent_dim=config.get('latent_dim', latent_dim),
                                     use_bottleneck=config.get('use_bottleneck', use_bottleneck),
                                     use_multiplicative_cond=config.get('use_multiplicative_cond', use_multiplicative_cond),
                                     )
        
        actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init['params']
        if 'batch_stats' in actor_def_init:
            actor_batch_stats = actor_def_init['batch_stats']
        else:
            actor_batch_stats = None
        
        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  tx=optax.adam(learning_rate=actor_lr),
                                  batch_stats=actor_batch_stats)
        
        self._rng = rng
        self._actor = actor
        
        network_def = StateActionEnsemble(hidden_dims, 
                                          num_qs=2, 
                                          use_action_sep=config.get('use_action_sep', use_action_sep), 
                                          use_normalized_features=config.get('use_normalized_features',use_normalized_features), 
                                          use_pixel_sep=config.get('use_pixel_sep', use_pixel_sep))
        
        critic_def_encoder = PixelMultiplexerEncoder(
                encoder=encoder_def,latent_dim=config.get('latent_dim', latent_dim), use_bottleneck=config.get('use_bottleneck',use_bottleneck),
                use_multiplicative_cond=config.get('use_multiplicative_cond',use_multiplicative_cond))
        
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

        critic_lr=0.0
        
        critic_encoder = TrainState.create(apply_fn=critic_def_encoder.apply,
                                   params=critic_encoder_params,
                                   batch_stats=critic_encoder_batch_stats,
                                   tx=optax.adam(learning_rate=critic_lr))
        
        critic_decoder = TrainState.create(apply_fn=critic_def_decoder.apply,
                                   params=critic_decoder_params,
                                   batch_stats=critic_decoder_batch_stats,
                                   tx=optax.adam(learning_rate=critic_lr))
        
        self._critic_encoder = critic_encoder
        self._critic_decoder = critic_decoder
        self._critic = (critic_encoder, critic_decoder)
        
        
        assert pathlib.Path(dir).is_file(), 'path {} not found!'.format(dir)
        sd = self._save_dict
        if 'actor' in sd:
            sd.pop('actor')
        output_dict = checkpoints.restore_checkpoint(dir, sd)
        self._critic_encoder, self._critic_decoder = output_dict['critic']
        self._critic=(self._critic_encoder, self._critic_decoder)
        

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, new_actor, info = _update_jit(self._rng, self._actor, self._critic_encoder, self._critic_decoder, batch, self.color_jitter)

        self._rng = new_rng
        self._actor = new_actor

        return info

    @property
    def _save_dict(self):
        save_dict = {
            'actor': self._actor,
            'critic': self._critic,
        }
        return save_dict

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).is_file(), 'path {} not found!'.format(dir)
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)

        self._actor = output_dict['actor']
        self._critic = output_dict['critic']
        (self._critic_encoder, self._critic_decoder) = output_dict['critic']
        print('restored from ', dir)

    def perform_eval(self, variant, i, wandb_logger, eval_buffer, eval_buffer_iterator, eval_env=None):
        num_batches = len(eval_buffer)// int(variant.batch_size)
        log_probs_list = []
        mse_list = []
        for _ in range(num_batches):
            test_batch = next(eval_buffer_iterator)
            log_probs = - self.eval_log_probs(test_batch)
            mse = self.eval_mse(test_batch)
            log_probs_list.append(log_probs)
            mse_list.append(mse)
        log_probs = sum(log_probs_list) / num_batches
        mse = sum(mse_list) / num_batches
        wandb_logger.log({f'evaluation/log_probs': log_probs}, step=i)
        wandb_logger.log({f'evaluation/MSE': mse}, step=i)

    def make_value_reward_visulization(self, variant, trajs):
        num_traj = len(trajs['rewards'])
        traj_images = []
        num_stack = variant.frame_stack
        for itraj in range(num_traj):
            observations = trajs['observations'][itraj]
            actions = trajs['actions'][itraj]
            rewards = trajs['rewards'][itraj]
            masks = trajs['masks'][itraj]

            # Do the frame stacking thing for observations
            images = np.lib.stride_tricks.sliding_window_view(observations.pop('pixels'),
                                                             num_stack + 1,
                                                             axis=0)

            for t in range(num_stack, len(actions)):
                action = actions[t]
                obs_pixels = images[t- num_stack]
                assert(t - num_stack) >= 0
                current_image = obs_pixels[..., :-1]
                next_image = obs_pixels[..., 1:]

                obs_dict = {'pixels': current_image}
                for k, v in observations.items():
                    obs_dict[k] = v[t]

            traj_images.append(make_visual(rewards, masks, images))
        print('finished reward value visuals.')
        return np.concatenate(traj_images, 0)


def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr

def make_visual(rewards, masks, images):

    fig, axs = plt.subplots(3, 1)
    canvas = FigureCanvas(fig)
    # plt.xlim([0, len(value_estimates)])

    assert len(images.shape) == 5
    images = images[..., -1]  # only taking the most recent image of the stack
    assert images.shape[-1] == 3

    interval = images.shape[0] // 4
    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)

    axs[0].imshow(sel_images)
    axs[1].plot(rewards, linestyle='--', marker='o')
    axs[1].set_ylabel('rewards')
    axs[1].set_xlim([0, len(rewards)])
    axs[2].plot(masks, linestyle='--', marker='d')
    axs[2].set_ylabel('boot masks')
    axs[2].set_xlim([0, len(masks)])

    plt.tight_layout()

    canvas.draw()  # draw the canvas, cache the renderer
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    # plt.show()
    return out_image

