"""Implementations of algorithms for continuous control."""
from jaxrl2.networks.learned_std_normal_policy import LearnedStdNormalPolicy
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

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from typing import Any

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.drq.augmentations import batched_random_crop, color_transform
from jaxrl2.agents.common import _unpack
from jaxrl2.agents.drq.drq_learner import _share_encoder
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.agents.iql.actor_updater import update_actor
from jaxrl2.agents.iql.critic_updater import update_q, update_v
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.normal_policy import NormalPolicy
from jaxrl2.networks.values import StateActionEnsemble, StateValue
from jaxrl2.networks.values.state_action_value import StateActionValue
from jaxrl2.networks.values.state_value import StateValueEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update
import wandb


class TrainState(train_state.TrainState):
    batch_stats: Any

@functools.partial(jax.jit, static_argnames=('critic_reduction', 'color_jitter', 'share_encoders', 'aug_next'))
def _update_jit(
    rng: PRNGKey, actor: TrainState, critic: TrainState,
    target_critic_params: Params, value: TrainState, batch: TrainState,
    discount: float, tau: float, expectile: float, A_scaling: float,
    critic_reduction: str, color_jitter: bool, share_encoders: bool, aug_next: bool
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str,
                                                                     float]]:
    batch = _unpack(batch)
    if share_encoders:
        actor = _share_encoder(source=critic, target=actor)

    rng, key = jax.random.split(rng)
    aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

    if color_jitter:
        rng, key = jax.random.split(rng)
        aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
    observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
    batch = batch.copy(add_or_replace={'observations': observations})

    if aug_next:
        rng, key = jax.random.split(rng)
        aug_next_pixels = batched_random_crop(key, batch['next_observations']['pixels'])
        if color_jitter:
            rng, key = jax.random.split(rng)
            aug_next_pixels = (color_transform(key, aug_next_pixels.astype(jnp.float32) / 255.) * 255).astype(jnp.uint8)
        next_observations = batch['next_observations'].copy(
            add_or_replace={'pixels': aug_next_pixels})
        batch = batch.copy(add_or_replace={'next_observations': next_observations})

    target_critic = critic.replace(params=target_critic_params)
    new_value, value_info = update_v(target_critic, value, batch, expectile,
                                     critic_reduction)
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, target_critic, new_value,
                                         batch, A_scaling, critic_reduction)

    new_critic, critic_info = update_q(critic, new_value, batch, discount)

    new_target_critic_params = soft_target_update(new_critic.params,
                                                  target_critic_params, tau)

    return rng, new_actor, new_critic, new_target_critic_params, new_value, {
        **critic_info,
        **value_info,
        **actor_info,
        'pixels': aug_pixels
    }


class PixelIQLLearner(Agent):

    def __init__(self,
                 seed: int,
                 observations: Union[jnp.ndarray, DatasetDict],
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.9,
                 A_scaling: float = 10.0,
                 critic_reduction: str = 'min',
                 dropout_rate: Optional[float] = None,
                 encoder_type='resnet_34_v1',
                 encoder_norm='batch',
                 policy_type='unit_std_normal',
                 policy_std=1.,
                 color_jitter = True,
                 share_encoders = False,
                 mlp_init_scale=1.,
                 mlp_output_scale=1.,
                 use_spatial_softmax=True,
                 softmax_temperature=1,
                 aug_next=False,
                 use_bottleneck=True
                 ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        self.aug_next=aug_next
        self.color_jitter = color_jitter
        self.share_encoders = share_encoders

        action_dim = actions.shape[-1]

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction
        self.A_scaling = A_scaling

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        if encoder_type == 'small':
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == 'impala':
            print('using impala')
            encoder_def = ImpalaEncoder()
            # encoder_def = SmallerImpalaEncoder()
        elif encoder_type == 'resnet_small':
            encoder_def = ResNetSmall(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_18_v1':
            encoder_def = ResNet18(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_34_v1':
            encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_small_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
        elif encoder_type == 'resnet_18_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif encoder_type == 'resnet_34_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        else:
            raise ValueError('encoder type not found!')

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if policy_type == 'unit_std_normal':
            policy_def = NormalPolicy(hidden_dims,
                                      action_dim,
                                      dropout_rate=dropout_rate,
                                      std=policy_std,
                                      init_scale=mlp_init_scale,
                                      output_scale=mlp_output_scale,
                                      )
        elif policy_type == 'learned_std_normal':
            policy_def = LearnedStdNormalPolicy(hidden_dims,
                                            action_dim,
                                            dropout_rate=dropout_rate)
        else:
            raise ValueError('policy type not found!')


        actor_def = PixelMultiplexer(encoder=encoder_def,
                                     network=policy_def,
                                     latent_dim=latent_dim,
                                     stop_gradient=share_encoders,
                                     use_bottleneck=use_bottleneck
                                     )
        actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init['params']
        actor_batch_stats = actor_def_init['batch_stats'] if 'batch_stats' in actor_def_init else None

        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  tx=optax.adam(learning_rate=actor_lr),
                                  batch_stats=actor_batch_stats)

        critic_def = StateActionEnsemble(hidden_dims, num_qs=2)
        critic_def = PixelMultiplexer(encoder=encoder_def,
                                      network=critic_def,
                                      latent_dim=latent_dim,
                                      use_bottleneck=use_bottleneck
                                      )
        critic_def_init = critic_def.init(critic_key, observations,
                                        actions)
        critic_params = critic_def_init['params']
        critic_batch_stats = critic_def_init['batch_stats'] if 'batch_stats' in critic_def_init else None
        critic = TrainState.create(apply_fn=critic_def.apply,
                                   params=critic_params,
                                   tx=optax.adam(learning_rate=critic_lr),
                                   batch_stats=critic_batch_stats
                                   )
        target_critic_params = copy.deepcopy(critic_params)

        value_def = StateValue(hidden_dims)
        value_def = PixelMultiplexer(encoder=encoder_def,
                                     network=value_def,
                                     latent_dim=latent_dim,
                                     use_bottleneck=use_bottleneck
                                     )
        value_def_init = value_def.init(value_key, observations)
        value_params = value_def_init['params']
        value_batch_stats = value_def_init['batch_stats'] if 'batch_stats' in value_def_init else None
        value = TrainState.create(apply_fn=value_def.apply,
                                  params=value_params,
                                  tx=optax.adam(learning_rate=value_lr),
                                  batch_stats=value_batch_stats
                                  )

        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._value = value

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, new_actor, new_critic, new_target_critic, new_value, info = _update_jit(
            self._rng, self._actor, self._critic, self._target_critic_params,
            self._value, batch, self.discount, self.tau, self.expectile,
            self.A_scaling, self.critic_reduction, self.color_jitter, self.share_encoders, self.aug_next
            )

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        self._value = new_value

        return info

    def perform_eval(self, variant, i, wandb_logger, eval_buffer, eval_buffer_iterator, eval_env):
        from examples.train_utils import make_multiple_value_reward_visulizations
        make_multiple_value_reward_visulizations(self, variant, i, eval_buffer, wandb_logger)

    def make_value_reward_visulization(self, variant, trajs):
        num_traj = len(trajs['rewards'])
        traj_images = []
        num_stack = variant.frame_stack
        for itraj in range(num_traj):
            observations = trajs['observations'][itraj]
            actions = trajs['actions'][itraj]
            rewards = trajs['rewards'][itraj]
            masks = trajs['masks'][itraj]

            q_pred = []
            v_pred = []

            # Do the frame stacking thing for observations
            images = np.lib.stride_tricks.sliding_window_view(observations.pop('pixels'),
                                                             num_stack + 1,
                                                             axis=0)

            for t in range(num_stack, len(actions)):
                action = actions[t][None]
                obs_pixels = images[t- num_stack]
                assert(t - num_stack) >= 0
                current_image = obs_pixels[..., :-1]
                next_image = obs_pixels[..., 1:]

                obs_dict = {'pixels': current_image[None]}
                for k, v in observations.items():
                    obs_dict[k] = v[t][None]

                value, q_value = get_value(action, obs_dict, self._value, self._critic)
                v_pred.append(value)
                q_pred.append(q_value)

            traj_images.append(make_visual(v_pred, q_pred, rewards, masks, images))
        print('finished reward value visuals.')
        return np.concatenate(traj_images, 0)

    @property
    def _save_dict(self):
        save_dict = {
            'value': self._value,
            'critic': self._critic,
            'target_critic_params': self._target_critic_params,
            'actor': self._actor
        }
        return save_dict

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).is_file(), 'path {} not found!'.format(dir)
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)

        self._actor = output_dict['actor']
        self._critic = output_dict['critic']
        self._value = output_dict['value']
        self._target_critic_params = output_dict['target_critic_params']
        print('restored from ', dir)


@functools.partial(jax.jit)
def get_value(action, observation, value, critic):
    input_collections = {'params': value.params}
    if value.batch_stats is not None:
        input_collections['batch_stats'] = value.batch_stats
    v_pred = value.apply_fn(input_collections, observation)

    input_collections = {'params': critic.params}
    if value.batch_stats is not None:
        input_collections['batch_stats'] = critic.batch_stats
    q_pred = critic.apply_fn(input_collections, observation, action)
    return v_pred, q_pred


def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr

def make_visual(value_estimates, q_estimates, rewards, masks, images):

    value_estimates_np = np.stack(value_estimates, 0).squeeze()
    q_estimates_np = np.stack(q_estimates, 0).squeeze()

    fig, axs = plt.subplots(5, 1)
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(value_estimates)])

    assert len(images.shape) == 5
    images = images[..., -1]  # only taking the most recent image of the stack
    assert images.shape[-1] == 3

    interval = images.shape[0] // 4
    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)

    axs[0].imshow(sel_images)
    axs[1].plot(value_estimates_np, linestyle='--', marker='o')
    axs[1].set_ylim([value_estimates_np.min(), value_estimates_np.max()])
    axs[1].set_ylabel('values')
    if len(q_estimates_np.shape) == 2:
        axs[2].plot(q_estimates_np[:, 0], linestyle='--', marker='o')
        axs[2].plot(q_estimates_np[:, 1], linestyle='--', marker='o')
    else:
        axs[2].plot(q_estimates_np, linestyle='--', marker='o')
    axs[2].set_ylabel('q values')
    axs[3].plot(rewards, linestyle='--', marker='o')
    axs[3].set_ylabel('rewards')
    axs[3].set_xlim([0, len(rewards)])
    if masks is not None:
        axs[4].plot(masks, linestyle='--', marker='d')
        axs[4].set_ylabel('boot masks')
        axs[4].set_xlim([0, len(masks)])

    plt.tight_layout()

    canvas.draw()  # draw the canvas, cache the renderer
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    # plt.show()
    return out_image