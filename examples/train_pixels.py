#! /usr/bin/env python
import copy
import sys

from jaxrl2.data.utils import get_task_id_mapping
from jaxrl2.utils.general_utils import AttrDict
import os

from jaxrl2.agents.cql.pixel_cql_learner import PixelCQLLearner
from jaxrl2.agents import PixelIQLLearner
from jaxrl2.agents import IQLLearner
from jaxrl2.agents.sac.sac_learner import SACLearner
from jaxrl2.agents.cql.pixel_cql_learner import PixelCQLLearner
from jaxrl2.agents.sarsa import PixelSARSALearner
from jaxrl2.agents.cql_parallel_overall.pixel_cql_learner import PixelCQLParallelLearner
from jaxrl2.agents.cql_encodersep.pixel_cql_learner import PixelCQLLearnerEncoderSep
from jaxrl2.agents.cql_encodersep_parallel.pixel_cql_learner import PixelCQLLearnerEncoderSepParallel

from jaxrl2.wrappers.prev_action_wrapper import PrevActionStack
from jaxrl2.wrappers.state_wrapper import StateStack
from jaxrl2.data.replay_buffer import ReplayBuffer
from jaxrl2.wrappers.reaching_reward_wrapper import ReachingReward
from jaxrl2.utils.general_utils import add_batch_dim
import collections
try:
    from doodad.wrappers.easy_launch import sweep_function, save_doodad_config
except:
    print('Warning, doodad not found!')
import time
from gym.spaces import Dict

import gym
import numpy as np
from tqdm import tqdm
from absl import app, flags
from gym.spaces import Box
from ml_collections import config_flags

from jaxrl2.data import MemoryEfficientReplayBuffer, MemoryEfficientReplayBufferParallel
from jaxrl2.data.dataset import MixingReplayBuffer, MixingReplayBufferParallel

from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
from examples.configs.dataset_config_sim import *

import roboverse
import wandb

from jaxrl2.wrappers import FrameStack, obs_latency
from examples.train_utils_sim import offline_training_loop, trajwise_alternating_training_loop, stepwise_alternating_training_loop, load_buffer
from jaxrl2.wrappers.reaching_reward_wrapper import compute_distance_reward
from jaxrl2.evaluation import evaluate
import argparse
import imp



TARGET_POINT = np.array([0.58996923,  0.21808016, -0.24382344])  # for reaching computed as the mean over all states.

def _process_image(obs):
    obs = (obs * 255).astype(np.uint8)
    obs = np.reshape(obs, (3, 128, 128))
    return np.transpose(obs, (1, 2, 0))


class Roboverse(gym.ObservationWrapper):

    def __init__(self, variant, env, num_tasks=1):
        super().__init__(env)

        # Note that the previous action is multiplied by FLAGS.frame_stack to
        # account for the ability to pass in multiple previous actions in the
        # system. This is the case especially when the number of previous actions
        # is supposed to be many, when using multiple past frames
        self.variant = variant
        obs_dict = {}
        if not variant.from_states:
            obs_dict['pixels'] = Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        if variant.add_states:
            obs_dict['state'] = Box(low=-100000, high=100000, shape=(10,), dtype=np.float32)
        if num_tasks > 1:
            obs_dict['task_id'] = Box(low=0, high=1, shape=(num_tasks,), dtype=np.float32)
        self.observation_space = Dict(obs_dict)

    def observation(self, observation):
        out_dict = {}
        if 'image' in observation:
            out_dict['pixels'] = _process_image(observation['image'])[None]
        if self.variant.add_states:
            out_dict['state'] = observation['state'][None]
        return out_dict


def main(variant):
    variant.stochastic_evals=False
    variant.cond_interfing=(variant.target_dataset in ['interfering'])
    
    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps

    expname = create_exp_name(variant.prefix, seed=variant.seed)
    outputdir = os.environ['EXP'] + '/jaxrl/' + expname
    variant.outputdir = outputdir
    print('writing to output dir ', outputdir)

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=outputdir, group_name=group_name)

    if variant.obs_latency:
        variant.dataset = variant.dataset + '_delay{}'.format(variant.obs_latency)

    train_tasks_neg = None
    if variant.dataset == 'ball_in_bowl':
        train_tasks = [put_ball_in_bowl]
        train_tasks_neg = [put_ball_in_bowl_neg]
    elif variant.dataset == 'ball_in_bowl_delay1':
        train_tasks = [put_ball_in_bowl_delay1]
    elif variant.dataset == 'multi_object_in_bowl':
        train_tasks, val_tasks = get_multi_object_in_bowl_data()
    elif variant.dataset == 'multi_object_in_bowl_interfering':
        train_tasks, val_tasks = get_multi_object_in_bowl_data_interfering()
    else:
        raise ValueError('dataset not found!')

    if variant.target_dataset != '':
        if variant.target_dataset == 'ball_in_bowl':
            target_train_tasks = [put_ball_in_bowl]
        elif variant.target_dataset == 'interfering':
            target_train_tasks = [interfering_task]
    else:
        target_train_tasks = []

    task_id_mapping = get_task_id_mapping(train_tasks + target_train_tasks, index = 3 if variant.cond_interfing else -3)
    print('task_id_mapping:', task_id_mapping)
    if variant.target_dataset != '':
        task_description = str.split(target_train_tasks[0], '/')[3 if variant.cond_interfing else -3]
        variant.eval_task_id = task_id_mapping[task_description]
        print('target task {} id {}'.format(task_description, variant.eval_task_id))
    num_tasks = len(task_id_mapping.keys())
    variant.num_tasks = num_tasks
    variant.task_id_mapping = task_id_mapping
    print('using {} tasks'.format(num_tasks))

    def wrap(env):
        if variant.reward_type != 'final_one':
            env = ReachingReward(env, TARGET_POINT, variant.reward_type)
        env = Roboverse(variant, env, num_tasks=num_tasks)
        if variant.obs_latency:
            env = obs_latency.ObsLatency(env, variant.obs_latency)
        if variant.add_prev_actions:
            if variant.frame_stack == 1:
                action_queuesize = 1
            else:
                action_queuesize = variant.frame_stack - 1
            env = PrevActionStack(env, action_queuesize)
        if variant.add_states:
            env = StateStack(env, variant.frame_stack)
        if not variant.from_states:
            env = FrameStack(env, variant.frame_stack)
        env = gym.wrappers.TimeLimit(env, 40)
        return env

    if variant.cond_interfing:
        env_name = 'PickPlaceInterferingDistractors-v0'
        extra_kwargs=dict(
            specific_task_id=True,
            desired_task_id=(1,0)
        )
        env = roboverse.make(env_name, transpose_image=True, **extra_kwargs)
    else:
        env_name = 'PutBallintoBowl-v0'
        extra_kwargs=dict()
        env = roboverse.make(env_name, transpose_image=True)
    env = wrap(env)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    print('seed', variant.seed)
    env.seed(variant.seed)
    if variant.from_states:
        env.disable_render()

    eval_env = roboverse.make(env_name, transpose_image=True, **extra_kwargs)
    eval_env = wrap(eval_env)
    eval_env.seed(variant.seed + 42)

    sample_obs = add_batch_dim(env.observation_space.sample())
    sample_action = add_batch_dim(env.action_space.sample())

    if variant.from_states:
        if variant.algorithm == 'iql':
            agent = IQLLearner(variant.seed, sample_obs,
                               sample_action, **kwargs)
        elif variant.algorithm == 'sac':
            agent = SACLearner(variant.seed, env.observation_space,
                               env.action_space, **kwargs)
    else:
        if variant.algorithm == 'iql':
            agent = PixelIQLLearner(variant.seed, sample_obs,
                                sample_action, **kwargs)
        elif variant.algorithm == 'cql':
            agent = PixelCQLLearner(variant.seed, sample_obs,
                                    sample_action, **kwargs)
        elif variant.algorithm == 'sarsa':
            agent = PixelSARSALearner(variant.seed, sample_obs,
                                    sample_action, **kwargs)
        elif variant.algorithm == 'cql_parallel':
            agent = PixelCQLParallelLearner(variant.seed, sample_obs,
                                            sample_action, **kwargs)
        elif variant.algorithm == 'cql_encodersep':
            agent = PixelCQLLearnerEncoderSep(variant.seed, sample_obs,
                                            sample_action, **kwargs)
        elif variant.algorithm == 'cql_encodersep_parallel':
            agent = PixelCQLLearnerEncoderSepParallel(variant.seed, sample_obs,
                                            sample_action, **kwargs)
    if variant.restore_path != '':
        agent.restore_checkpoint(variant.restore_path)
    if variant.from_states:
        replay_buffer_class = ReplayBuffer
    else:
        if variant.algorithm in ['cql_encodersep_parallel', 'cql_parallel']:
            if variant.target_dataset != '':
                replay_buffer_class = MemoryEfficientReplayBuffer
            else:
                replay_buffer_class = MemoryEfficientReplayBufferParallel
        else:
            replay_buffer_class = MemoryEfficientReplayBuffer
    
    if variant.algorithm in ['cql_encodersep_parallel', 'cql_parallel']:
        mixing_buff_class = MixingReplayBufferParallel
    else:
        mixing_buff_class = MixingReplayBuffer      

    if not variant.online_from_scratch:
        all_trajs_pos, buffer_size = load_mult_tasks(variant, train_tasks)
        print('size positives', buffer_size)
        if variant.use_negatives:
            all_trajs_neg, buffer_size_neg = load_mult_tasks(variant,train_tasks_neg)
            print('size neg', buffer_size_neg)
        else:
            buffer_size_neg, all_trajs_neg = 0, []

        replay_buffer = replay_buffer_class(
            env.observation_space, env.action_space,
            buffer_size + buffer_size_neg)

        insert_data(variant, replay_buffer, all_trajs_pos + all_trajs_neg, task_id_mapping=task_id_mapping)

        if variant.target_dataset != '':
            trajs, num_transitions = load_mult_tasks(variant, target_train_tasks, num_traj_cutoff=variant.num_target_traj)
            print('size target', num_transitions)
            replay_buffer_target = replay_buffer_class(env.observation_space, env.action_space, num_transitions)
            insert_data(variant, replay_buffer_target, trajs, task_id_mapping=task_id_mapping)
            replay_buffer = mixing_buff_class([replay_buffer, replay_buffer_target], variant.target_mixing_ratio)

        offline_training_loop(variant, agent, eval_env, replay_buffer, None,  wandb_logger)

        if variant.algorithm in ['cql_encodersep_parallel', 'cql_parallel']:
            # reset buffers to be non parallel
            # when going to online fintuning make sure only the outer most buffer is parallel type
            # pretty ineficient but only need to do once
            if variant.target_dataset != '':
                replay_buffer = MixingReplayBuffer([replay_buffer, replay_buffer_target], variant.target_mixing_ratio)
            else:
                del replay_buffer
                all_trajs_pos, buffer_size = load_mult_tasks(variant, train_tasks)
                print('size positives', buffer_size)
                if variant.use_negatives:
                    all_trajs_neg, buffer_size_neg = load_mult_tasks(variant,train_tasks_neg)
                    print('size neg', buffer_size_neg)
                else:
                    buffer_size_neg, all_trajs_neg = 0, []

                replay_buffer = MemoryEfficientReplayBuffer(
                    env.observation_space, env.action_space,
                    buffer_size + buffer_size_neg)

                insert_data(variant, replay_buffer, all_trajs_pos + all_trajs_neg, task_id_mapping=task_id_mapping)

    if variant.cql_alpha_online_finetuning > 0:
        agent._cql_alpha = variant.cql_alpha_online_finetuning

    online_replay_buffer = replay_buffer_class(env.observation_space, env.action_space, int(5e5))
    if not variant.online_from_scratch:
        replay_buffer = mixing_buff_class([replay_buffer, online_replay_buffer], variant.online_mixing_ratio)
        print('buffer len', replay_buffer.length())
    else:
        replay_buffer = online_replay_buffer

    if variant.restore_path != '':
        restore_folder = '/'.join(str.split(variant.restore_path, '/')[:-1])
        if 'replaybuffer.npy' in os.listdir(restore_folder):
            online_replay_buffer.restore(restore_folder + '/replaybuffer.npy')
        print("restored replay buffer")

    replay_buffer.seed(variant.seed)

    if variant.trajwise_alternating:
        trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer,
                                           wandb_logger, real_env=False)
    else:
        assert not variant.only_add_success
        stepwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, outputdir, replay_buffer, wandb_logger)


def load_mult_tasks(variant, train_tasks, num_traj_cutoff=None):
    buffer_size = 0
    all_trajs_pos = []
    for dataset_file in train_tasks:
        num_transitions, trajs = load_buffer(dataset_file, variant, num_traj_cutoff=num_traj_cutoff)
        buffer_size += num_transitions
        all_trajs_pos.extend(trajs)
    return all_trajs_pos, buffer_size


def insert_data(variant, replay_buffer, trajs, run_test=False, task_id_mapping=None):
    for traj_id, traj in enumerate(trajs):
        if variant.frame_stack == 1:
            action_queuesize = 1
        else:
            action_queuesize = variant.frame_stack - 1
        prev_actions = collections.deque(maxlen=action_queuesize)
        current_states = collections.deque(maxlen=variant.frame_stack)

        for i in range(action_queuesize):
            prev_action = np.zeros_like(traj['actions'][0])
            if run_test:
                prev_action[0] = -1
            prev_actions.append(prev_action)

        for i in range(variant.frame_stack):
            state = traj['observations'][0]['state']
            if run_test:
                state[0] = 0
            current_states.append(state)

        for i in range(len(traj['observations'])):
            obs = dict()
            if not variant.from_states:
                obs['pixels'] = traj['observations'][i]['image']
                obs['pixels'] = obs['pixels'][..., np.newaxis]
                if run_test:
                    obs['pixels'][0, 0] = i
            if variant.add_states:
                obs['state'] = np.stack(current_states, axis=-1)
            if variant.add_prev_actions:
                obs['prev_action'] = np.stack(prev_actions, axis=-1)

            action_i = traj['actions'][i]
            if run_test:
                action_i[0] = i
            prev_actions.append(action_i)

            current_state = traj['next_observations'][i]['state']
            if run_test:
                current_state[0] = i + 1
            current_states.append(current_state)  # do not delay state, therefore use i instead of i

            next_obs = dict()
            if not variant.from_states:
                next_obs['pixels'] = traj['next_observations'][i]['image']
                next_obs['pixels'] = next_obs['pixels'][..., np.newaxis]
                if i == 0:
                    obs['pixels'] = np.tile(obs['pixels'], [1, 1, 1, variant.frame_stack])
                if run_test:
                    next_obs['pixels'][0, 0] = i + 1
            if variant.add_states:
                next_obs['state'] = np.stack(current_states, axis=-1)
            if variant.add_prev_actions:
                next_obs['prev_action'] = np.stack(prev_actions, axis=-1)

            if variant.reward_type != 'final_one':
                reward = compute_distance_reward(traj['observations'][i]['state'][:3], TARGET_POINT, variant.reward_type)
            else:
                reward = traj['rewards'][i]
            reward = reward * variant.reward_scale + variant.reward_shift
            if run_test:
                reward = i

            if traj['rewards'][i]:
                mask = 0.0
            else:
                mask = 1.0

            if run_test:
                mask = i

            if task_id_mapping is not None:
                if len(task_id_mapping.keys()) > 1:
                    task_id = np.zeros((len(task_id_mapping.keys())))
                    task_id[task_id_mapping[traj['task_description']]] = 1
                    obs['task_id'] = task_id
                    next_obs['task_id'] = task_id

            replay_buffer.insert(
                dict(observations=obs,
                     actions=traj['actions'][i],
                     next_actions=traj['actions'][i+1] if len(traj['actions']) > i+1 else traj['actions'][i],
                     rewards=reward,
                     next_observations=next_obs,
                     masks=mask,
                     dones=bool(i == len(traj['observations']) - 1),
                     trajectory_id=replay_buffer._traj_counter,
                     ))

        replay_buffer.increment_traj_counter()
