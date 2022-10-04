"""Implementations of algorithms for continuous control."""
import numpy as np
import matplotlib
matplotlib.use('Agg')

import pathlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

from flax.training import checkpoints
import copy
import functools
from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.iql.actor_updater import update_actor
from jaxrl2.agents.iql.critic_updater import update_q, update_v
from jaxrl2.networks import NormalPolicy
from jaxrl2.networks.values import StateActionEnsemble, StateValue
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update
from jaxrl2.networks.learned_std_normal_policy import LearnedStdNormalPolicy


@functools.partial(jax.jit, static_argnames='critic_reduction')
def _update_jit(
    rng: PRNGKey, actor: TrainState, critic: TrainState,
    target_critic_params: Params, value: TrainState, batch: TrainState,
    discount: float, tau: float, expectile: float, A_scaling: float,
    critic_reduction: str
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str,
                                                                     float]]:

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
        **actor_info
    }


class IQLLearner(Agent):

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.9,
                 A_scaling: float = 10.0,
                 critic_reduction: str = 'min',
                 dropout_rate: Optional[float] = None,
                 policy_type='unit_std_normal',
                 policy_std=1.,
                 share_encoders=True,
                 ensemble_over_value=False,
                 mlp_init_scale=1.,
                 mlp_output_scale=1.
                 ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        assert not ensemble_over_value  #todo

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction
        self.A_scaling = A_scaling

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        if policy_type == 'unit_std_normal':
            actor_def = NormalPolicy(hidden_dims,
                                     action_dim,
                                     dropout_rate=dropout_rate,
                                     std=policy_std,
                                     init_scale=mlp_init_scale,
                                     output_scale=mlp_output_scale,
                                     )
        elif policy_type == 'learned_std_normal':
            actor_def = LearnedStdNormalPolicy(hidden_dims,
                                            action_dim,
                                            dropout_rate=dropout_rate)
        else:
            raise ValueError('policy type not found!')

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        actor_params = actor_def.init(actor_key, observations)['params']
        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  tx=optax.adam(learning_rate=actor_lr))

        critic_def = StateActionEnsemble(hidden_dims, num_qs=2)
        critic_params = critic_def.init(critic_key, observations,
                                        actions)['params']
        critic = TrainState.create(apply_fn=critic_def.apply,
                                   params=critic_params,
                                   tx=optax.adam(learning_rate=critic_lr))
        target_critic_params = copy.deepcopy(critic_params)

        value_def = StateValue(hidden_dims)
        value_params = value_def.init(value_key, observations)['params']
        value = TrainState.create(apply_fn=value_def.apply,
                                  params=value_params,
                                  tx=optax.adam(learning_rate=value_lr))

        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._value = value

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, new_actor, new_critic, new_target_critic, new_value, info = _update_jit(
            self._rng, self._actor, self._critic, self._target_critic_params,
            self._value, batch, self.discount, self.tau, self.expectile,
            self.A_scaling, self.critic_reduction)

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        self._value = new_value

        return info


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

            for t in range(num_stack, len(actions)):
                action = actions[t]
                assert(t - num_stack) >= 0

                obs_dict = {}
                for k, v in observations.items():
                    obs_dict[k] = v[t]

                value, q_value = get_value(action, obs_dict, self._value, self._critic)
                v_pred.append(value)
                q_pred.append(q_value)

            traj_images.append(make_visual(v_pred, q_pred, rewards, masks))
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
    v_pred = value.apply_fn({'params': value.params}, observation)
    q_pred = critic.apply_fn({'params': critic.params}, observation, action)
    return v_pred, q_pred


def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr

def make_visual(value_estimates, q_estimates, rewards, masks):

    value_estimates_np = np.stack(value_estimates, 0)
    q_estimates_np = np.stack(q_estimates, 0)

    fig, axs = plt.subplots(4, 1)
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(value_estimates)])

    axs[0].plot(value_estimates_np, linestyle='--', marker='o')
    axs[0].set_ylim([value_estimates_np.min(), value_estimates_np.max()])
    axs[0].set_ylabel('values')
    axs[1].plot(q_estimates_np[:, 0], linestyle='--', marker='o')
    axs[1].plot(q_estimates_np[:, 1], linestyle='--', marker='o')
    axs[1].set_ylabel('q values')
    axs[2].plot(rewards, linestyle='--', marker='o')
    axs[2].set_ylabel('rewards')
    axs[2].set_xlim([0, len(rewards)])
    axs[3].plot(masks, linestyle='--', marker='d')
    axs[3].set_ylabel('boot masks')
    axs[3].set_xlim([0, len(masks)])

    plt.tight_layout()

    canvas.draw()  # draw the canvas, cache the renderer
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    # plt.show()
    return out_image
