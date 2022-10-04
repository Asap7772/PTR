#! /usr/bin/env python
import os
import time

import gym
import numpy as np
import tensorflow as tf
import tqdm
from flax.metrics.tensorboard import SummaryWriter
from gym.spaces import Box

from jaxrl2.data import MemoryEfficientReplayBuffer

from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import pickle

import roboverse
import argparse
import wandb

# import jaxrl2.extra_envs.dm_control_suite
from jaxrl2.agents import PixelIQLLearner
from jaxrl2.evaluation import evaluate
from doodad.wrappers.easy_launch import sweep_function, save_doodad_config
import ml_collections
from ml_collections.config_dict import config_dict
from jaxrl2.wrappers import FrameStack, wrap_pixels

def _process_image(obs):
    obs = (obs * 255).astype(np.uint8)
    obs = np.reshape(obs, (3, 128, 128))
    return np.transpose(obs, (1, 2, 0))


class Roboverse(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0,
                                     high=255,
                                     shape=(128, 128, 3),
                                     dtype=np.uint8)

    def observation(self, observation):
        return _process_image(observation)

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

def main(args, config): 
    summary_writer = SummaryWriter(os.path.join(args.save_dir, 'tb'))
    def wrap(env):
        env = TakeKey(env, take_key='image')
        env = Roboverse(env)
        if args.add_time:
            env = AddTime(env)
        env = FrameStack(env, 1)  # This codebase always assumes stacking.
        env = gym.wrappers.TimeLimit(env, 25)
        return env

    env_name = 'PutBallintoBowl-v0'
    env = roboverse.make(env_name, transpose_image=True)
    env = wrap(env)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(args.seed)

    eval_env = roboverse.make(env_name, transpose_image=True)
    eval_env = wrap(eval_env)
    eval_env.seed(args.seed + 42)

    kwargs = dict(config)
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = args.max_steps
    variant = {}

    for attr, flag_obj in vars(args).items():
        variant[attr] = flag_obj

    variant['train_kwargs'] = kwargs

    expname = create_exp_name(args.prefix, seed=args.seed)
    outputdir = os.environ['EXP'] + '/jaxrl/' + expname
    print('writing to output dir ', outputdir)

    # if args.seed != 0:
    #     wait_time = np.random.randint(0, 180)
    #     print('waiting for {} seconds to avoid problems with launch'.format(wait_time))
    #     time.sleep(wait_time)
    group_name = args.prefix + args.launch_group_id
    wandb_logger = WandBLogger(args.prefix != '', variant, 'iql_sim', experiment_id=expname, output_dir=outputdir, group_name=group_name)

    agent = PixelIQLLearner(args.seed, env.observation_space.sample(),
                            env.action_space.sample(), **kwargs)

    dataset_file = os.path.join(os.environ['DATA'], 'minibullet',
                                'pickplacedata_noise0.1',
                                env_name, 'train', 'out.npy')
    replay_buffer_size, trajs_pos = load_buffer(dataset_file)
    print('size positives', replay_buffer_size)

    if args.use_negatives:
        dataset_file_negatives = os.path.join(
            os.environ['DATA'], 'minibullet',
            'pickplacedata_noise0.1_failonly250', env_name,
            'train', 'out.npy')
        replay_buffer_size_neg, trajs_neg = load_buffer(dataset_file_negatives)
        print('size neg', replay_buffer_size_neg)
    else:
        replay_buffer_size_neg, trajs_neg = 0, None

    replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space,
        replay_buffer_size + replay_buffer_size_neg)

    insert_data(args, replay_buffer, trajs_pos)

    if args.use_negatives:
        insert_data(replay_buffer, trajs_neg, nofinal_bootstrap=args.negatives_nofinal_bootstrap)

    replay_buffer.seed(args.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(args.batch_size)

    for i in tqdm.tqdm(range(1, args.max_steps + 1),
                       smoothing=0.1,
                       disable=not args.tqdm):
        batch = next(replay_buffer_iterator)
        update_info = agent.update(batch)

        if i % args.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.scalar(f'training/{k}', v, i)
                    wandb_logger.log({f'training/{k}': v}, step=i)
                else:
                    summary_writer.histogram(f'training/{k}', v, i)
            summary_writer.flush()

        if i % args.eval_interval == 0:
            eval_info = evaluate(agent,
                                 eval_env,
                                 num_episodes=args.eval_episodes)
            # video = eval_info['obs'].transpose(0, 3, 1, 2)
            # wandb_logger.log({'eval_video': wandb.Video(video[:, -3:], fps=8)}, step=i)

            for k, v in eval_info.items():
                if v.ndim == 0:
                    summary_writer.scalar(f'evaluation/{k}', v, i)
                    wandb_logger.log({f'evaluation/{k}': v}, step=i)

            trajs = replay_buffer.get_random_trajs(3)
            images = agent.make_value_reward_visulization(trajs)
            wandb_logger.log({'reward_value_images' : wandb.Image(images)}, step=i)


def insert_data(args, replay_buffer, trajs, nofinal_bootstrap=False):
    for traj_id, traj in enumerate(trajs):
        for i in range(len(traj['observations'])):
            if traj['terminals'][i]:
                mask = 0.0
            else:
                mask = 1.0

            if nofinal_bootstrap:
                if i == len(traj['observations']) - 1:
                    mask = 0.0

            reward = traj['rewards'][i] * args.reward_scale + args.reward_shift

            obs = traj['observations'][i]['image']
            if args.add_time:
                obs = _add_time(obs, i)[..., np.newaxis]
            else:
                obs = obs[..., np.newaxis]

            next_obs = traj['next_observations'][i]['image']
            if args.add_time:
                next_obs = _add_time(next_obs, i + 1)[..., np.newaxis]
            else:
                next_obs = next_obs[..., np.newaxis]


            replay_buffer.insert(
                dict(observations=obs,
                     actions=traj['actions'][i],
                     rewards=reward,
                     next_observations=next_obs,
                     masks=mask,
                     dones=traj['terminals'][i],
                     trajectory_id=replay_buffer._traj_counter
                     ))

        replay_buffer.increment_traj_counter()



def load_buffer(dataset_file):
    trajs = np.load(dataset_file, allow_pickle=True)
    replay_buffer_size = 0
    for traj in trajs:
        for i in range(len(traj['observations'])):
            replay_buffer_size += 1
    return replay_buffer_size, trajs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default=os.environ['EXP'], help="path to tensorboard logging")
    parser.add_argument("--seed", default=42, help="Random seed.")
    parser.add_argument("--eval_episodes", default=10, help="Number of episodes used for evaluation.")
    parser.add_argument("--log_interval", default=1000, help="Logging interval.")
    parser.add_argument("--eval_interval", default=5000, help="Eval interval.")
    parser.add_argument("--batch_size", default=256, help="Mini batch size.")
    parser.add_argument("--max_steps", default=int(5e5), help="Number of training steps.")
    parser.add_argument("--tqdm", action='store_true', help="use tqdm progress bar")
    parser.add_argument("--save_video", action='store_true', help="Save videos during evaluation.")
    parser.add_argument("--use_negatives", action='store_true', help="Use negative_data")
    parser.add_argument("--add_time", action='store_true', help="add time conditioning")
    parser.add_argument("--reward_scale", default=1.0, help="Scale for the reward")
    parser.add_argument("--reward_shift", default=0.0, help="Shift for the reward")
    parser.add_argument("--prefix", help="experiment prefix, if given creates subfolder in experiment directory", required=True)
    parser.add_argument("--launch_group_id", default='', help='group id used to group runs on wandb.')
    parser.add_argument("--negatives_nofinal_bootstrap", action='store_true',  help="apply bootstrapping at last time step of negatives")
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, args.prefix)

    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.value_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.cnn_features = (16, 32, 64, 128, 256)
    config.cnn_strides = (2, 2, 2, 2, 2)
    config.cnn_padding = 'VALID'
    config.latent_dim = 50

    config.discount = 0.99

    config.expectile = 0.7  # The actual tau for expectiles.
    config.A_scaling = 1.0
    config.dropout_rate = config_dict.placeholder(float)
    config.cosine_decay = True

    config.tau = 0.005

    config.critic_reduction = 'mean'

    # app.run(main)

    def train(doodad_config, default_params):
        main(args, config)
        save_doodad_config(doodad_config)

    params_to_sweep = {}
    
    sweep_function(
        train,
        params_to_sweep,
        default_params={},
        log_path=args.prefix,
        mode='here_no_doodad',
        use_gpu=True,
        num_gpu=1,
    )


