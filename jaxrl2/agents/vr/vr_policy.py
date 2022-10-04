"""Implementations of algorithms for continuous control."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
import functools
from typing import Dict, Optional, Sequence, Tuple, Union
from jaxrl2.data.dataset import DatasetDict

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from typing import Any

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.drq.augmentations import batched_random_crop, color_transform
from jaxrl2.agents.common import _unpack
from jaxrl2.types import PRNGKey


from widowx_envs.control_loops import Environment_Exception
import widowx_envs.utils.transformation_utils as tr
from pyquaternion import Quaternion
from transformations import quaternion_from_matrix

import rospy
import tf2_ros
import geometry_msgs.msg
import random
import time

def publish_transform(transform, name):
    translation = transform[:3, 3]

    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = 'wx250s/base_link'
    t.child_frame_id = name
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]

    quat = quaternion_from_matrix(transform)
    t.transform.rotation.w = quat[0]
    t.transform.rotation.x = quat[1]
    t.transform.rotation.y = quat[2]
    t.transform.rotation.z = quat[3]

    # print('publish transofrm', name)
    br.sendTransform(t)

class TrainState(train_state.TrainState):
    batch_stats: Any

@functools.partial(jax.jit)
def _update_jit(
    rng: PRNGKey, batch: TrainState,
) -> Tuple[PRNGKey, Dict[str, float]]:
    batch = _unpack(batch)

    rng, key1, key2 = jax.random.split(rng, num=3)
    aug_pixels = batched_random_crop(key1, batch['observations']['pixels'])
    aug_pixels = (color_transform(key2, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
    observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
    batch = batch.copy(add_or_replace={'observations': observations})

    rng, key = jax.random.split(rng)

    return rng, {'pixels': aug_pixels}

class VRPolicy(Agent):

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
                 **kwargs
                 ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        self.action_dim = actions.shape[-1]
        self.last_pressed_times = {}
        self.env = None
        self.prev_handle_press = False
        self.reference_vr_transform = None
        self.reference_robot_transform = None
        self.internal_counter = 0
        self.internal_counter_default_policy = 0
        rng = jax.random.PRNGKey(seed)
        self._rng = rng
    
    def set_env(self, env):
        self.env = env
        self.reader = self.env.oculus_reader

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, info = _update_jit(self._rng, batch)
        self._rng = new_rng
        return info

    @property
    def _save_dict(self):
        save_dict = {
            'actor': self._actor,
        }
        return save_dict

    def perform_eval(self, variant, i, wandb_logger, eval_buffer, eval_buffer_iterator, eval_env=None, do_control_eval=False):
        pass

    def get_default_action(self, observations: np.ndarray) -> np.ndarray:
        actions = np.zeros(self.action_dim)
        actions[-1] = 1
        return actions

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        return self.eval_actions(observations)

    def get_pose_and_button(self):
        poses, buttons = self.reader.get_transformations_and_buttons()
        if poses == {}:
            return None, None, None, None
        return poses['r'], buttons['RTr'], buttons['rightTrig'][0], buttons['RG']

    def oculus_to_robot(self, current_vr_transform):
        current_vr_transform = tr.RpToTrans(Quaternion(axis=[0, 0, 1], angle=-np.pi / 2).rotation_matrix,
                                            np.zeros(3)).dot(
            tr.RpToTrans(Quaternion(axis=[1, 0, 0], angle=np.pi / 2).rotation_matrix, np.zeros(3))).dot(
            current_vr_transform)
        return current_vr_transform

    def reset(self):
        self.internal_counter = 0
        self.internal_counter_default_policy = 0
        self.prev_vr_transform = None  # used for self.act_use_deltas only

        # used for act_use_fixed_reference only:
        self.prev_handle_press = False
        self.reference_vr_transform = None
        self.reference_robot_transform = None

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        
        self.last_update_time = time.time()
        t1 = time.time()
        current_vr_transform, trigger, trigger_continuous, handle_button = self.get_pose_and_button()
        if current_vr_transform is None:
            return self.get_default_action(observations)
        else:
            if not self.prev_handle_press and handle_button:
                print("resetting reference pose")
                self.internal_counter_default_policy = 0
                self.reference_vr_transform = self.oculus_to_robot(current_vr_transform)
                self.initial_vr_offset = tr.RpToTrans(np.eye(3), self.reference_vr_transform[:3, 3])
                self.reference_vr_transform = tr.TransInv(self.initial_vr_offset).dot(self.reference_vr_transform)  ##

                self.reference_robot_transform, _ = self.env.get_target_state()
                self.prev_commanded_transform = self.reference_robot_transform

            if not handle_button:
                self.internal_counter = 0
                self.internal_counter_default_policy += 1
                self.reference_vr_transform = None
                self.reference_robot_transform, _ = self.env.get_target_state()
                self.prev_handle_press = False
                self.prev_commanded_transform = self.reference_robot_transform
                return self.get_default_action(observations)
        self.prev_handle_press = True
        self.internal_counter += 1

        current_vr_transform = self.oculus_to_robot(current_vr_transform)
        current_vr_transform = tr.TransInv(self.initial_vr_offset).dot(current_vr_transform)  ##

        publish_transform(current_vr_transform, 'currentvr_robotsystem')
        delta_vr_transform = current_vr_transform.dot(tr.TransInv(self.reference_vr_transform))

        publish_transform(self.reference_robot_transform, 'reference_robot_transform')
        M_rob, p_rob = tr.TransToRp(self.reference_robot_transform)
        M_delta, p_delta = tr.TransToRp(delta_vr_transform)
        new_robot_transform = tr.RpToTrans(M_delta.dot(M_rob), p_rob + p_delta)

        publish_transform(new_robot_transform, 'des_robot_transform')

        prev_target_pos, _ = self.env.get_target_state()
        delta_robot_transform = new_robot_transform.dot(tr.TransInv(prev_target_pos))
        publish_transform(delta_robot_transform, 'delta_robot_transform')
        self.prev_commanded_transform = new_robot_transform

        des_gripper_position = (1 - trigger_continuous)
        # import pdb; pdb.set_trace()
        actions = tr.transform2action_local(delta_robot_transform, des_gripper_position, self.env.controller.get_cartesian_pose()[:3])

        if np.linalg.norm(actions[:3]) > 0.2:
            print('delta transform too large! Press c and enter to continue')
            import pdb; pdb.set_trace()
            raise Environment_Exception

        return actions


class VRPolicyDAgger(VRPolicy):
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
                 ):
        super(VRPolicyDAgger, self).__init__(seed, observations, actions, actor_lr, decay_steps, hidden_dims, 
                                             cnn_features, cnn_strides, cnn_padding, latent_dim, dropout_rate, 
                                             encoder_type, encoder_norm, use_spatial_softmax, softmax_temperature)
        self.default_policy = None
        self.all_task_ids = None
        self.task_idx_string = None
        self.policy_T = 50
        self.default_policy_T = 50
    
    def set_hps(self, default_policy=None, all_task_ids=None, task_idx_string=None, policy_T=50, default_policy_T=50):
        self.default_policy = default_policy
        self.policy_T = policy_T
        self.default_policy_T = default_policy_T
        self.all_task_ids = all_task_ids
        self.task_idx_string = task_idx_string
    
    def reset(self):
        self.task_id = random.choice(self.all_task_ids)
        self.policy_desc = self.task_idx_string[self.task_id]
        print("Sampled task ", self.policy_desc)
        return super().reset()

    def get_default_action(self, observations: np.ndarray) -> np.ndarray:
        action = self.default_policy.eval_actions(observations)
        return action

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        print('Human counter ', self.internal_counter)
        print('Robot counter ', self.internal_counter_default_policy)
        print('Sampled task ', self.policy_desc)
        actions = super(VRPolicyDAgger, self).eval_actions(observations)
        return actions
