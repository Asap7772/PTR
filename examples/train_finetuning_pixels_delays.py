#! /usr/bin/env python
import os

import gym
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from gym.spaces import Box
from ml_collections import config_flags
from gym.spaces import Dict

from jaxrl2.data import MemoryEfficientReplayBuffer
from jaxrl2.data.dataset import MixingReplayBuffer

from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name, create_group_name

tf.config.experimental.set_visible_devices([], "GPU")
import pickle

import roboverse

# import jaxrl2.extra_envs.dm_control_suite
from jaxrl2.agents import PixelCQLLearner
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import FrameStack, wrap_pixels, obs_latency, PrevActionStack, StateStack

import wandb

import collections

FLAGS = flags.FLAGS


flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(5e5), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('use_negatives', True, 'Use negative_data')
flags.DEFINE_float('reward_scale', 1.0, 'Scale for the reward')
flags.DEFINE_float('reward_shift', 0.0, 'Shift for the reward')
flags.DEFINE_float('cql_alpha', 1.0, 'Alpha for CQL')
flags.DEFINE_float('discount', 0.99, 'Discount factor to use')
flags.DEFINE_boolean('max_q_backup', False, 'Max Q-backup to use')
flags.DEFINE_boolean('use_impala_for_critic', False, 'Use IMPALA for critic')
flags.DEFINE_boolean('use_impala_for_policy', False, 'Use IMPALA for policy')
flags.DEFINE_integer('online_start', 200000, 'When to start online finetuning')
flags.DEFINE_float('online_online_mixing_ratio', 0.5, 'What ratio of offline/online samples to play')

flags.DEFINE_integer('obs_latency', 0, 'Number of timesteps of observation latency')
flags.DEFINE_integer('frame_stack', 1, 'Number of frames stacked')
flags.DEFINE_boolean('add_states', False, 'whether to add low-dim states to the obervations')
flags.DEFINE_boolean('add_prev_actions', False, 'whether to add low-dim previous actions to the obervations')

flags.DEFINE_string('wandb_project', 'cql_sim_latency', 'wandb project')

flags.DEFINE_string('prefix', '', 'prefix to use for wandb')
config_flags.DEFINE_config_file(
    'config',
    'examples/configs/offline_pixels_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def _process_image(obs):
    obs = (obs * 255).astype(np.uint8)
    obs = np.reshape(obs, (3, 128, 128))
    return np.transpose(obs, (1, 2, 0))


class Roboverse(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        # Note that the previous action is multiplied by FLAGS.frame_stack to 
        # account for the ability to pass in multiple previous actions in the
        # system. This is the case especially when the number of previous actions
        # is supposed to be many, when using multiple past frames
        self.observation_space = Dict(
            dict(pixels=Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
                 state=Box(low=-100000, high=100000, shape=(10,), dtype=np.float32)))

    def observation(self, observation):
        return {
            'pixels': _process_image(observation['image']),
            'state': observation['state'],
        } 

def _add_time(obs, t):
    return np.pad(obs,
                  pad_width=((0, 0), (0, 0), (1, 0)),
                  mode='constant',
                  constant_values=t)


class AddTime(gym.ObservationWrapper):

    def __init__(self, env, increment=8):
        super().__init__(env)
        low = env.observation_space.low
        high = env.observation_space.high
        assert low.ndim == 3 and high.ndim == 3
        assert env.observation_space.dtype == np.uint8
        low = _add_time(low, 0)
        high = _add_time(high, 255)
        self.observation_space = Box(low=low, high=high, dtype=np.uint8)
        self._increment = increment

    def reset(self):
        self._t = 0
        return super().reset()

    def step(self, action):
        self._t += self._increment
        assert self._t < 256
        return super().step(action)

    def observation(self, observation):
        return _add_time(observation, self._t)


def main(_):
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb'))

    def wrap(env):
        env = Roboverse(env)
        if FLAGS.add_prev_actions:
            # Only do this to the extent that there is one less frame to be
            # stacked, since this only concats the previous actions, not the
            # current action....
            env = PrevActionStack(env, FLAGS.frame_stack - 1)
        if FLAGS.add_states:
            env = StateStack(env, FLAGS.frame_stack)
        env = FrameStack(env, FLAGS.frame_stack)
        if FLAGS.obs_latency:
            env = obs_latency.ObsLatency(env, FLAGS.obs_latency)
        env = gym.wrappers.TimeLimit(env, 40)
        return env

    env = roboverse.make('PutBallintoBowl-v0', transpose_image=True)
    env = wrap(env)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)

    eval_env = env

    variant = {}
    for attr, flag_obj in FLAGS.__flags.items():
        variant[attr] = flag_obj.value

    kwargs = dict(FLAGS.config)
    kwargs['cql_alpha'] = FLAGS.cql_alpha
    kwargs['discount'] = FLAGS.discount
    kwargs['max_q_backup'] = FLAGS.max_q_backup
    kwargs['use_impala_for_critic'] = FLAGS.use_impala_for_critic
    kwargs['use_impala_for_policy'] = FLAGS.use_impala_for_policy

    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = FLAGS.max_steps

    variant['train_kwargs'] = kwargs

    expname = create_exp_name(FLAGS.prefix)
    outputdir = '/nfs/kun2/users/aviralkumar/' + '/jaxrl_bridge_data_with_delays/' + expname
    print('writing to output dir ', outputdir)

    group_name = create_group_name(FLAGS.prefix)
    wandb_logger = WandBLogger(FLAGS.prefix != '', variant,
                        FLAGS.wandb_project,
                        experiment_id=expname, output_dir=outputdir, group_name=group_name)

    # Update kwargs to account for the prev action and the state
    kwargs['use_prev_action'] = FLAGS.add_prev_actions
    kwargs['use_proprio_state'] = FLAGS.add_states

    agent = PixelCQLLearner(FLAGS.seed, env.observation_space.sample(),
                            env.action_space.sample(), **kwargs)

    dataset_file = os.path.join('/nfs/kun1/users/febert/data/minibullet/pickplacedata_noise0.1_run_until_end',
                                'PutBallintoBowl-v0', 'train', 'out.npy')

    replay_buffer_size, trajs_pos = load_buffer(dataset_file)
    print('size positives', replay_buffer_size)

    if FLAGS.use_negatives:
        dataset_file_negatives = os.path.join(
            '/nfs/kun1/users/febert/data/minibullet/pickplacedata_noise0.1_failonly_run_until_end', 'PutBallintoBowl-v0',
            'train', 'out.npy')
        replay_buffer_size_neg, trajs_neg = load_buffer(dataset_file_negatives)
        print('size neg', replay_buffer_size_neg)
    else:
        replay_buffer_size_neg, trajs_neg = 0, None

    replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space,
        replay_buffer_size + replay_buffer_size_neg)

    insert_data(replay_buffer, trajs_pos)

    if FLAGS.use_negatives:
        insert_data(replay_buffer, trajs_neg)

    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(FLAGS.batch_size)

    for i in tqdm.tqdm(range(1, FLAGS.online_start + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = next(replay_buffer_iterator)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.scalar(f'offline_training/{k}', v, i)
                    wandb_logger.log({f'offline_training/{k}': v}, step=i)
                else:
                    summary_writer.histogram(f'offline_training/{k}', v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent,
                                 eval_env,
                                 num_episodes=FLAGS.eval_episodes)

            video= np.stack([ts['pixels'] for ts in eval_info.pop('obs')])
            video = video[..., -1].transpose(0, 3, 1, 2)
            wandb_logger.log({'eval_video': wandb.Video(video, fps=8)}, step=i)

            for k, v in eval_info.items():
                if v.ndim == 0:
                    summary_writer.scalar(f'evaluation/{k}', v, i)
                    wandb_logger.log({f'evaluation/{k}': v}, step=i)
            
            trajs = replay_buffer.get_random_trajs(3)
            images = agent.make_value_reward_visulization_with_delays(trajs, FLAGS.frame_stack)
            wandb_logger.log({'reward_value_images' : wandb.Image(images)}, step=i)

    # Make replay buffers
    online_replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space,
        int(5e5)
    )
    finetuning_replay_buffer = MixingReplayBuffer(
        [replay_buffer, online_replay_buffer], 
        [int(FLAGS.batch_size * FLAGS.online_mixing_ratio), FLAGS.batch_size - int(FLAGS.batch_size * FLAGS.online_mixing_ratio)]
    )
    print('buffer len', finetuning_replay_buffer.length())

    finetuning_replay_buffer.seed(FLAGS.seed)
    finetuning_replay_buffer_iterator = finetuning_replay_buffer.get_iterator(FLAGS.batch_size)


    # Finetuning setting now.
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(FLAGS.online_start + 1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        
        if len(finetuning_replay_buffer.replay_buffers[1]) > 1000:
            batch = next(finetuning_replay_buffer_iterator)
            update_info = agent.update(batch)
        
        if done:
            print ('Trajectory ended: ', current_len)
            observation, done = env.reset(), False
            online_replay_buffer.increment_traj_counter()
            current_len = 0

        action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        reward = reward * FLAGS.reward_scale + FLAGS.reward_shift

        online_replay_buffer.insert(
            dict(observations=observation,
                    actions=action,
                    rewards=reward,
                    masks=mask,
                    dones=done,
                    next_observations=next_observation,
                    time_step=current_len,
                    trajectory_id=online_replay_buffer._traj_counter))
        
        observation = next_observation
        current_len += 1

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    wandb_logger.log({f'online_training/{k}': v}, step=i)
                else:
                    wandb_logger.log_histogram(f'online_training/{k}', v, i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent,
                                 eval_env,
                                 num_episodes=FLAGS.eval_episodes)
            video= np.stack([ts['pixels'] for ts in eval_info.pop('obs')])
            video = video[..., -1].transpose(0, 3, 1, 2)
            wandb_logger.log({'eval_video': wandb.Video(video[:, -3:], fps=8)}, step=i)

            for k, v in eval_info.items():
                if v.ndim == 0:
                    wandb_logger.log({f'evaluation/{k}': v}, step=i)

            trajs = finetuning_replay_buffer.get_random_trajs(3)
            if isinstance(trajs, list):
                images = agent.make_value_reward_visulization_with_delays(trajs[0], FLAGS.frame_stack)
                wandb_logger.log({'reward_value_images_offline' : wandb.Image(images)}, step=i)
                images = agent.make_value_reward_visulization_with_delays(trajs[1], FLAGS.frame_stack)
                wandb_logger.log({'reward_value_images_online' : wandb.Image(images)}, step=i)
            else:
                images = agent.make_value_reward_visulization(trajs)
                wandb_logger.log({'reward_value_images' : wandb.Image(images)}, step=i)

 
def insert_data(replay_buffer, trajs):

    for traj_id, traj in enumerate(trajs):

        next_transitions_are_redundant = False
        prev_actions = collections.deque(maxlen=FLAGS.frame_stack-1)
        current_states = collections.deque(maxlen=FLAGS.frame_stack)

        for i in range(FLAGS.frame_stack-1):
            prev_actions.append(np.zeros_like(traj['actions'][0]))

        for i in range(FLAGS.frame_stack):
            current_states.append(traj['observations'][0]['state'])

        for i in range(len(traj['observations'])):

            # Don't add remaining transitions of the trajectory if the reward of 1
            # is seen atleast once.
            if next_transitions_are_redundant:
                pass
            else:                
                if FLAGS.obs_latency:
                    assert FLAGS.obs_latency > 0
                    i_obs = i - FLAGS.obs_latency
                    i_obs = np.max([i_obs, 0])
                else:
                    i_obs = i

                current_states.append(traj['observations'][i_obs]['state'])

                obs = dict()
                obs['pixels'] = traj['observations'][i_obs]['image']
                obs['pixels'] = obs['pixels'][..., np.newaxis]
                if FLAGS.add_states:
                    obs['state'] = np.stack(current_states, axis=-1)
                if FLAGS.frame_stack > 1 and FLAGS.add_prev_actions:
                    obs['prev_action'] = np.stack(prev_actions, axis=-1)

                prev_actions.append(traj['actions'][i])

                next_obs = dict()
                next_obs['pixels'] = traj['next_observations'][i_obs]['image']
                next_obs['pixels'] = next_obs['pixels'][..., np.newaxis]
                if FLAGS.add_states:
                    next_obs['state'] = np.stack(current_states, axis=-1)
                if FLAGS.frame_stack > 1 and FLAGS.add_prev_actions:
                    next_obs['prev_action'] = np.stack(prev_actions, axis=-1)

                if i == 0:
                    obs['pixels'] = np.tile(obs['pixels'], [1, 1, 1, FLAGS.frame_stack])

                reward = traj['rewards'][i_obs] * FLAGS.reward_scale + FLAGS.reward_shift

                if traj['terminals'][i_obs]:
                    mask = 0.0
                else:
                    mask = 1.0

                replay_buffer.insert(
                    dict(observations=obs,
                        actions=traj['actions'][i],
                        rewards=reward,
                        next_observations=next_obs,
                        masks=mask,
                        dones=traj['terminals'][i_obs], # terminals also need to change to be delayed, since they go hand in hand with reward
                        trajectory_id=replay_buffer._traj_counter,
                ))

                # The following means that the remaining transitions are redudant and we were
                # simply waiting for it to get to a 1 reward... but lets not put the remaining
                # transitions in the buffer
                if traj['rewards'][i_obs] > 0:
                    next_transitions_are_redundant = True

        replay_buffer.increment_traj_counter()


def load_buffer(dataset_file):
    trajs = np.load(dataset_file, allow_pickle=True)
    replay_buffer_size = 0
    for traj in trajs:
        for i in range(len(traj['observations'])):
            replay_buffer_size += 1
    return replay_buffer_size, trajs


if __name__ == '__main__':
    app.run(main)
