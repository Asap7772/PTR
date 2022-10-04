#! /usr/bin/env python
import os

import gym
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from ml_collections import config_flags

tf.config.experimental.set_visible_devices([], "GPU")

import jaxrl2.extra_envs.dm_control_suite
from jaxrl2.agents import SACLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    'configs/sac_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb'))

    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=False)
    eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    agent = SACLearner(FLAGS.seed, env.observation_space, env.action_space,
                       **kwargs)

    replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                 FLAGS.max_steps)
    replay_buffer.seed(FLAGS.seed)

    returns = []
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(observations=observation,
                 actions=action,
                 rewards=reward,
                 masks=mask,
                 dones=done,
                 next_observations=next_observation))
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info['episode'].items():
                decode = {'r': 'return', 'l': 'length', 't': 'time'}
                summary_writer.scalar(f'training/{decode[k]}', v, i)

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.scalar(f'training/{k}', v, i)
                summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent,
                                 eval_env,
                                 num_episodes=FLAGS.eval_episodes)
            for k, v in eval_info.items():
                summary_writer.scalar(f'evaluation/{k}', v, i)
            returns.append((i, eval_info['return']))
            savefile = os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt')
            np.savetxt(savefile, returns, fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
