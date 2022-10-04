"""Implementations of algorithms for continuous control."""
import matplotlib
matplotlib.use('Agg')
from flax.training import checkpoints
import pathlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
import functools
from typing import Dict, Optional, Sequence, Tuple, Union
from jaxrl2.data.dataset import DatasetDict

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from typing import Any

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.drq.augmentations import batched_random_crop, color_transform
from jaxrl2.agents.common import _unpack
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.agents.bc.bc_learner import _reward_loss_update_jit
from jaxrl2.networks.normal_policy import NormalPolicy
from jaxrl2.types import Params, PRNGKey

class TrainState(train_state.TrainState):
    batch_stats: Any

@functools.partial(jax.jit, static_argnames=('color_jitter'))
def _update_jit(
    rng: PRNGKey, actor: TrainState, batch: TrainState, color_jitter: bool
) -> Tuple[PRNGKey, TrainState, Dict[str, float]]:
    batch = _unpack(batch)

    rng, key1, key2 = jax.random.split(rng, num=3)
    aug_pixels = batched_random_crop(key1, batch['observations']['pixels'])
    if color_jitter:
        aug_pixels = (color_transform(key2, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
    observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
    batch = batch.copy(add_or_replace={'observations': observations})

    rng, key = jax.random.split(rng)
    _, new_actor, actor_info = _reward_loss_update_jit(key, actor, batch)

    return rng, new_actor,{**actor_info, 'pixels': aug_pixels}


class PixelRewardLearner(Agent):

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
                 encoder_type='resnet_34_v1',
                 encoder_norm='batch',
                 use_spatial_softmax=True,
                 softmax_temperature=1.0,
                 use_bottleneck=True,
                 color_jitter = True,
                 ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        self.color_jitter = color_jitter

        action_dim = actions.shape[-1]
        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng, 2)

        if encoder_type == 'small':
            print('using small encoder')
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == 'impala':
            print('using impala')
            encoder_def = ImpalaEncoder()
            # encoder_def = SmallerImpalaEncoder()
        elif encoder_type == 'resnet_18_v1':
            print('using resnet18v1')
            encoder_def = ResNet18(norm=encoder_norm)
        elif encoder_type == 'resnet_34_v1':
            print('using resnet34v1')
            encoder_def = ResNet34(norm=encoder_norm)
        elif encoder_type == 'resnet_small_v2':
            print('using impalav2')
            encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1),norm=encoder_norm)
        elif encoder_type == 'resnet_18_v2':
            print('using resnet18v2')
            encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif encoder_type == 'resnet_34_v2':
            print(f'using resnet34v2 with {encoder_norm} norm')
            encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        else:
            raise ValueError('encoder type not found!')

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)
        
        policy_def = NormalPolicy(hidden_dims,
                                    action_dim=1,
                                    dropout_rate=dropout_rate)

        actor_def = PixelMultiplexer(encoder=encoder_def,
                                     network=policy_def,
                                     latent_dim=latent_dim,
                                     use_bottleneck=use_bottleneck,)
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

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, new_actor, info = _update_jit(
            self._rng, self._actor, batch, self.color_jitter)

        self._rng = new_rng
        self._actor = new_actor

        return info

    @property
    def _save_dict(self):
        save_dict = {
            'actor': self._actor,
        }
        return save_dict

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).is_file(), 'path {} not found!'.format(dir)
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)

        self._actor = output_dict['actor']
        print('restored from ', dir)

    def perform_eval(self, variant, i, wandb_logger, eval_buffer, eval_buffer_iterator, eval_env=None):
        num_batches = len(eval_buffer)// int(variant.batch_size)
        reward_loss_list = []
        for _ in range(num_batches):
            test_batch = next(eval_buffer_iterator)
            reward_loss = self.eval_reward_function(test_batch)
            reward_loss_list.append(reward_loss)
        reward_loss = sum(reward_loss_list) / num_batches
        wandb_logger.log({f'evaluation/reward loss': reward_loss}, step=i)

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

