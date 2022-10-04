"""Implementations of algorithms for continuous control."""

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flax.training import checkpoints
import pathlib

import copy
import functools
from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state

from jaxrl2.agents.agent import Agent

from jaxrl2.agents.sarsa.actor_updater import update_actor
from jaxrl2.agents.sarsa.mu_updater import update_mu
from jaxrl2.agents.sarsa.critic_updater import update_critic
from jaxrl2.agents.sarsa.temperature_updater import update_temperature
from jaxrl2.agents.sarsa.temperature import Temperature
from jaxrl2.utils.target_update import soft_target_update
from jaxrl2.types import Params, PRNGKey

from jaxrl2.agents.agent import Agent
from jaxrl2.networks.learned_std_normal_policy import LearnedStdNormalPolicy, LearnedStdTanhNormalPolicy
from jaxrl2.agents.drq.augmentations import batched_random_crop, color_transform
from jaxrl2.agents.common import _unpack
from jaxrl2.agents.drq.drq_learner import _share_encoder
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.normal_policy import NormalPolicy
from jaxrl2.networks.values import StateActionEnsemble, StateValue
from jaxrl2.networks.values.state_action_value import StateActionValue
from jaxrl2.networks.values.state_value import StateValueEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update
import wandb

import numpy as np
from typing import Any

class TrainState(train_state.TrainState):
    batch_stats: Any = None

@functools.partial(jax.jit, static_argnames=['critic_reduction', 'backup_entropy', 'max_q_backup', 'method', 'method_type', 'cross_norm', 'color_jitter'])
def _update_jit(
    rng: PRNGKey, actor: TrainState, critic: TrainState,
    target_critic_params: Params, temp: TrainState, batch: TrainState,
    discount: float, tau: float, target_entropy: float, backup_entropy: bool,
    critic_reduction: str, cql_alpha: float, max_q_backup: bool, dr3_coefficient: float,
    method:bool = False, method_const:float = 0.0, method_type:int=0, 
    cross_norm:bool = False, color_jitter:bool = False,
    ) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str,float]]:

    batch = _unpack(batch)

    rng, key = jax.random.split(rng)
    
    aug_pixels = batched_random_crop(key, batch['observations']['pixels'])
    if color_jitter:
        rng, key = jax.random.split(rng)
        aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

    observations = batch['observations'].copy(
        add_or_replace={'pixels': aug_pixels})
    batch = batch.copy(add_or_replace={'observations': observations})

    key, rng = jax.random.split(rng)
    
    if True:
        target_critic = critic.replace(params=target_critic_params)
        new_critic, critic_info = update_critic(
            key,
            actor,
            critic,
            target_critic,
            temp,
            batch,
            discount,
            backup_entropy=backup_entropy,
            critic_reduction=critic_reduction,
            cql_alpha=cql_alpha,
            max_q_backup=max_q_backup,
            dr3_coefficient=dr3_coefficient,
            method=method,
            method_const=method_const,
            method_type=method_type,
            cross_norm=cross_norm,
        )
        new_target_critic_params = soft_target_update(new_critic.params, target_critic_params, tau)
    else:
        new_critic, critic_info, new_target_critic_params = critic, {}, target_critic_params

    if False:
        rng, key = jax.random.split(rng)
        new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch, cross_norm=cross_norm)
        new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)
    else:
        new_actor, actor_info, new_temp, alpha_info = actor, {}, temp, {}

    return rng, new_actor, new_critic, new_target_critic_params, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }

class PixelSARSALearner(Agent):

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
                 latent_dim: int = 50,
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
                 use_spatial_softmax=True,
                 softmax_temperature=-1,
                 share_encoders=False,
                 color_jitter=True,
                 use_bottleneck=True,
                 **kwargs,
        ):
        print('unused kwargs:', kwargs)

        self.color_jitter=color_jitter

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim
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

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        if encoder_type == 'small':
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == 'impala':
            print('using impala')
            encoder_def = ImpalaEncoder()
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

        
        policy_def = LearnedStdTanhNormalPolicy(hidden_dims,action_dim, dropout_rate=dropout_rate)

        actor_def = PixelMultiplexer(encoder=policy_encoder_def,
                                     network=policy_def,
                                     latent_dim=latent_dim,
                                     stop_gradient=share_encoders,
                                     use_bottleneck=use_bottleneck)

        actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init['params']
        actor_batch_stats = actor_def_init['batch_stats'] if 'batch_stats' in actor_def_init else None
        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  batch_stats=actor_batch_stats,
                                  tx=optax.adam(learning_rate=actor_lr))

        critic_def = StateActionEnsemble(hidden_dims, num_qs=2)
        critic_def = PixelMultiplexer(encoder=encoder_def,
                                      network=critic_def,
                                      latent_dim=latent_dim,
                                      use_bottleneck=use_bottleneck)

        critic_def_init = critic_def.init(critic_key, observations, actions)
        critic_params = critic_def_init['params']
        critic_batch_stats = critic_def_init['batch_stats'] if 'batch_stats' in critic_def_init else None

        critic = TrainState.create(apply_fn=critic_def.apply,
                                   params=critic_params,
                                   batch_stats=critic_batch_stats,
                                   tx=optax.adam(learning_rate=critic_lr))

        target_critic_params = copy.deepcopy(critic_params)

        
        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                                 params=temp_params,
                                 tx=optax.adam(learning_rate=temp_lr))

        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._temp = temp
        self._target_critic_params = target_critic_params
        self._cql_alpha = cql_alpha
        print ('Discount: ', self.discount)
        print ('CQL Alpha: ', self._cql_alpha)
        print('Method: ', self.method, 'Const: ', self.method_const)

    
    def update(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, new_actor, new_critic, new_target_critic_params, new_temp, info = _update_jit(
            self._rng, self._actor, self._critic, self._target_critic_params,
            self._temp, batch, self.discount, self.tau, self.target_entropy,
            self.backup_entropy, self.critic_reduction, self._cql_alpha, self.max_q_backup,
            self.dr3_coefficient, color_jitter=self.color_jitter, method=self.method,
            method_const=self.method_const, method_type=self.method_type, cross_norm=self.cross_norm)

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic_params
        self._temp = new_temp

        return info

    def perform_eval(self, variant, i, wandb_logger, eval_buffer, eval_buffer_iterator, eval_env):
        trajs = eval_buffer.get_random_trajs(3)
        if isinstance(trajs, list):
            images = self.make_value_reward_visulization(variant, trajs[0])
            wandb_logger.log({'reward_value_images_offline': wandb.Image(images)}, step=i)
            images = self.make_value_reward_visulization(variant, trajs[1])
            wandb_logger.log({'reward_value_images_online': wandb.Image(images)}, step=i)
        else:
            images = self.make_value_reward_visulization(variant, trajs)
            wandb_logger.log({'reward_value_images': wandb.Image(images)}, step=i)

    def make_value_reward_visulization(self, variant, trajs, **kwargs):
        # try:
        num_traj = len(trajs['rewards'])
        traj_images = []
        num_stack = variant.frame_stack
        for itraj in range(num_traj):
            observations = trajs['observations'][itraj]
            actions = trajs['actions'][itraj]
            rewards = trajs['rewards'][itraj]
            masks = trajs['masks'][itraj]

            q_pred = []
            # Do the frame stacking thing for observations
            images = np.lib.stride_tricks.sliding_window_view(observations.pop('pixels'), num_stack + 1, axis=0)

            for t in range(num_stack, len(actions)):
                action = actions[t][None]
                obs_pixels = images[t - num_stack]
                assert (t - num_stack) >= 0
                current_image = obs_pixels[..., :-1]
                next_image = obs_pixels[..., 1:]

                obs_dict = {'pixels': current_image[None]}
                for k, v in observations.items():
                    obs_dict[k] = v[t][None]
                q_value = get_q_value(action, obs_dict, self._critic)
                q_pred.append(q_value)

            traj_images.append(self.make_visual(q_pred, rewards, images, masks))
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
            'temp': self._temp
        }
        return save_dict

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).is_file(), 'path {} not found!'.format(dir)
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)

        self._actor = output_dict['actor']
        self._critic = output_dict['critic']
        self._target_critic_params = output_dict['target_critic_params']
        self._temp = output_dict['temp']

        print('restored from ', dir)


    def make_visual(self, q_estimates, rewards, images, masks=None, show_window=False):
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

        plt.tight_layout()

        canvas.draw()
        out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if show_window:
            plt.imshow(out_image)
            plt.show()
        plt.close(fig)
        return out_image

    def show_value_reward_visulization(self, traj):
        q_vals = []
        for step in traj:
            q_vals.append(get_q_value(step['action'], step['observation'], self._critic))
        rewards = [step['reward'] for step in traj]
        images = np.stack([step['observation']['pixels'][0] for step in traj], 0)
        self.make_visual(q_vals, rewards, images, show_window=True)

@functools.partial(jax.jit)
def get_q_value(actions, obs_dict, critic):
    if critic.batch_stats is not None:
        q_pred, _ = critic.apply_fn({'params': critic.params, 'batch_stats': critic.batch_stats}, obs_dict, actions, mutable=['batch_stats'])
    else:
        q_pred = critic.apply_fn({'params': critic.params}, obs_dict, actions)
    return q_pred

def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr