#! /usr/bin/env python
import os

import gym
import tensorflow as tf
import tqdm
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from ml_collections import config_flags

tf.config.experimental.set_visible_devices([], "GPU")
from jaxrl2.agents import BCLearner, IQLLearner
from jaxrl2.data import D4RLDataset
from jaxrl2.evaluation import evaluate, evaluate_log_prob
from jaxrl2.wrappers import CDFNormalizer, wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    'configs/offline_config.py:bc',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb'))

    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env)
    env.seed(FLAGS.seed)

    dataset = D4RLDataset(env)
    dataset.seed(FLAGS.seed)

    if 'antmaze' in FLAGS.env_name:
        if hasattr(FLAGS.config.model_config, 'reward_shift'):
            dataset.dataset_dict[
                'rewards'] += FLAGS.config.model_config.reward_shift
            dataset.dataset_dict[
                'rewards'] *= FLAGS.config.model_config.reward_scale
            del FLAGS.config.model_config.reward_shift
            del FLAGS.config.model_config.reward_scale

    if (hasattr(FLAGS.config.model_config, 'distr')
            and FLAGS.config.model_config.distr == 'ar'):
        env = CDFNormalizer(env,
                            dataset.dataset_dict['actions'],
                            update_actions_inplace=True)

    train_dataset, test_dataset = dataset, None

    kwargs = dict(FLAGS.config.model_config)
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = FLAGS.max_steps
    agent = globals()[FLAGS.config.model_constructor](
        FLAGS.seed, env.observation_space.sample(), env.action_space.sample(),
        **kwargs)

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = train_dataset.sample(FLAGS.batch_size)
        info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in info.items():
                if v.ndim == 0:
                    summary_writer.scalar(f'training/{k}', v, i)
                else:
                    summary_writer.histogram(f'training/{k}', v, i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, env, num_episodes=FLAGS.eval_episodes)
            eval_info['return'] = env.get_normalized_score(eval_info['return'])
            for k, v in eval_info.items():
                summary_writer.scalar(f'evaluation/{k}', v, i)

            train_log_prob = evaluate_log_prob(agent, train_dataset)
            summary_writer.scalar(f'training/log_prob', train_log_prob, i)

            if test_dataset is not None:
                test_log_prob = evaluate_log_prob(agent, test_dataset)
                summary_writer.scalar(f'evaluation/log_prob', test_log_prob, i)


if __name__ == '__main__':
    app.run(main)
