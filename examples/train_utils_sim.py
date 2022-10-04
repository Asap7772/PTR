import copy
import os

import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np
import jax.numpy as jnp
import wandb
from jaxrl2.evaluation import evaluate
import collections
from jaxrl2.utils.visualization_utils import visualize_image_actions
from jaxrl2.wrappers.reaching_reward_wrapper import compute_distance_reward
from jaxrl2.utils.visualization_utils import visualize_states_rewards, visualize_image_rewards
from jaxrl2.data.dataset import MixingReplayBuffer
from jaxrl2.utils.visualization_utils import sigmoid

def offline_training_loop(variant, agent, eval_env, replay_buffer, eval_replay_buffer=None, wandb_logger=None, perform_control_evals=True, task_id_mapping=None):
    if eval_replay_buffer is None:
        eval_replay_buffer = replay_buffer
    
    changed_buffer_to_finetuning = False
    if isinstance(replay_buffer, MixingReplayBuffer):
        replay_buffer.set_mixing_ratio(1) #offline only    
    
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if eval_replay_buffer is not None:
        eval_replay_buffer_iterator = eval_replay_buffer.get_iterator(variant.batch_size)
    for i in tqdm(range(1, variant.online_start + 1), smoothing=0.1,):
        
        if not changed_buffer_to_finetuning and i >= variant.offline_finetuning_start and isinstance(replay_buffer, MixingReplayBuffer):
            replay_buffer.set_mixing_ratio(variant.target_mixing_ratio)
            del replay_buffer_iterator
            replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
            changed_buffer_to_finetuning = True
            
            if variant.cql_alpha_offline_finetuning > 0:
                agent._cql_alpha = variant.cql_alpha_offline_finetuning
            
        t0 = time.time()
        batch = next(replay_buffer_iterator)
        tget_data = time.time() - t0
        t1 = time.time()
        update_info = agent.update(batch)
        tupdate = time.time() - t1

        if i % variant.eval_interval == 0:
            if hasattr(agent, 'unreplicate'):
                agent.unreplicate()
            wandb_logger.log({'t_get_data': tget_data}, step=i)
            wandb_logger.log({'t_update': tupdate}, step=i)
            if 'pixels' in update_info and i % (variant.eval_interval*10) == 0:
                if variant.algorithm == 'reward_classifier':
                    image = visualize_image_rewards(update_info.pop('pixels'), batch['rewards'], update_info.pop('rewards_mean'), batch['observations'], task_id_mapping=task_id_mapping)
                    wandb_logger.log({'training/image_rewards': wandb.Image(image)}, step=i)
                else:
                    image = visualize_image_actions(update_info.pop('pixels'), batch['actions'], update_info.pop('pred_actions_mean'))
                    wandb_logger.log({'training/image_actions': wandb.Image(image)}, step=i)
            if perform_control_evals:
                perform_control_eval(agent, eval_env, i, variant, wandb_logger)
            agent.perform_eval(variant, i, wandb_logger, eval_replay_buffer, eval_replay_buffer_iterator, eval_env)
            if hasattr(agent, 'replicate'):
                agent.replicate()

        if i % variant.log_interval == 0:
            for k, v in update_info.items():
                
                if v.ndim == 0:
                    wandb_logger.log({f'training/{k}': v}, step=i)
                elif v.ndim <= 2:
                    wandb_logger.log_histogram(f'training/{k}', v, i)

        if variant.checkpoint_interval != -1:
            if i % variant.checkpoint_interval == 0:
                if hasattr(agent, 'unreplicate'):
                    agent.unreplicate()
                agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)
                if hasattr(agent, 'replicate'):
                    agent.replicate()

def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       perform_control_evals=True, real_env=True, saver=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)

    traj_collect_func = collect_traj

    traj_id = 0

    i = variant.online_start + 1
    with tqdm(total=variant.max_steps + 1) as pbar:
        while i < variant.max_steps + 1:

            traj = traj_collect_func(variant, agent, env, not variant.stochastic_data_collect, traj_id=traj_id)
            traj_id += 1

            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            if saver is not None:
                saver.save(traj)
            print('collecting traj len online buffer', len(online_replay_buffer))

            if len(online_replay_buffer) > variant.start_online_updates:
                for _ in range(len(traj)*variant.multi_grad_step):
                    # online perform update once we have some amount of online trajs
                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch)
                    pbar.update()
                    i += 1

                    if i % variant.log_interval == 0:
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        wandb_logger.log({'replay_buffer_size': len(online_replay_buffer)}, i)

                    if i % variant.eval_interval == 0:
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger)
                        agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    if variant.checkpoint_interval != -1:
                        if i % variant.checkpoint_interval == 0:
                            agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)
                            if hasattr(variant, 'save_replay_buffer') and variant.save_replay_buffer:
                                print('saving replay buffer to ', variant.outputdir + '/replaybuffer.npy')
                                online_replay_buffer.save(variant.outputdir + '/replaybuffer.npy')

def add_online_data_to_buffer(variant, traj, online_replay_buffer):
    if variant.only_add_success:
        if traj[-1]['reward'] < 1e-3:
            print('trajecotry discarded because unsuccessful')
            return

    online_replay_buffer.increment_traj_counter()
    for step in traj:
        if not step['done'] or 'TimeLimit.truncated' in step['info'] or 'Error.truncated' in step['info']:
            mask = 1.0
        else:
            mask = 0.0
        reward = step['reward'] * variant.reward_scale + variant.reward_shift

        obs = step['observation']
        next_obs = step['next_observation']
        if not variant.add_states and 'state' in obs:
            obs.pop('state')
        if not variant.add_states and 'state' in next_obs:
            next_obs.pop('state')

        online_replay_buffer.insert(
            dict(observations=obs,
                 actions=step['action'],
                 rewards=reward,
                 masks=mask,
                 dones=step['done'],
                 next_observations=next_obs,
                 trajectory_id=online_replay_buffer._traj_counter
                 ))

def run_multiple_trajs(variant, agent, env, num_trajs, deterministic=True):
    returns = []
    lengths = []
    obs = []

    traj_collect_func = collect_traj

    for i in range(num_trajs):
        print('##############################################')
        print('traj', i)
        traj = traj_collect_func(variant, agent, env, deterministic, traj_id=i)
        returns.append(np.sum([step['reward'] for step in traj]))
        lengths.append(len(traj))
        obs.append([step['observation'] for step in traj])

    return {
        'return': np.mean(returns),
        'length': np.mean(lengths),
        'obs': obs[-1],
        'rewards': np.array([step['reward'] for step in traj])
    }

def collect_traj(variant, agent, env, deterministic, traj_id=None):
    obs, done = env.reset(), False
    traj = []

    print('collect traj deterministc', deterministic)
    while not done:
        if hasattr(variant, 'eval_task_id'):
            if variant.eval_task_id != -1:
                obs['task_id'] = np.zeros(variant.num_tasks, np.float32)[None]
                obs['task_id'][:, variant.eval_task_id] = 1.
        if variant.from_states:
            obs_filtered = copy.deepcopy(obs)
            if 'pixels' in obs_filtered:
                obs_filtered.pop('pixels')
        else:
            obs_filtered = obs

        if deterministic:
            action = agent.eval_actions(obs_filtered).squeeze()
        else:
            action = agent.sample_actions(obs_filtered).squeeze()
        next_obs, reward, done, info = env.step(action)

        if hasattr(variant, 'eval_task_id'):
            if variant.eval_task_id != -1:
                next_obs['task_id'] = obs['task_id']

        traj.append({
            'observation': obs,
            'action': action,
            'reward': reward,
            'next_observation': next_obs,
            'done': done,
            'info': info
        })
        obs = next_obs
    return traj


def stepwise_alternating_training_loop(variant, batch_size, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger):
    replay_buffer_iterator = replay_buffer.get_iterator(batch_size)
    observation, done = env.reset(), False
    print('stepwise alternating loop')
    for i in tqdm(range(variant.online_start + 1, variant.max_steps + 1),
                       smoothing=0.1,
                       ):

        if len(replay_buffer.replay_buffers[1]) > variant.start_online_updates:
            batch = next(replay_buffer_iterator)
            update_info = agent.update(batch)
            print('gradient update')

        if done:
            observation, done = env.reset(), False
            online_replay_buffer.increment_traj_counter()

        action = agent.eval_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        reward = reward * variant.reward_scale + variant.reward_shift

        online_replay_buffer.insert(
            dict(observations=observation,
                 actions=action,
                 rewards=reward,
                 masks=mask,
                 dones=done,
                 next_observations=next_observation,
                 trajectory_id=online_replay_buffer._traj_counter
                 ))
        observation = next_observation

        if i % variant.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    wandb_logger.log({f'training/{k}': v}, step=i)
                elif v.ndim <= 2:
                    wandb_logger.log_histogram(f'training/{k}', v, i)

        if i % variant.eval_interval == 0:
            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

        if variant.checkpoint_interval != -1:
            if i % variant.checkpoint_interval == 0:
                agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)


def perform_eval(variant, agent, eval_env, i, replay_buffer, wandb_logger, do_control_eval=True):
    if do_control_eval:
        perform_control_eval(agent, eval_env, i, variant, wandb_logger)
    trajs = replay_buffer.get_random_trajs(3)
    if not hasattr(variant, 'target_dataset') or variant.target_dataset == '':
        if isinstance(trajs, list):
                images = agent.make_value_reward_visulization(variant, trajs[0])
                wandb_logger.log({'reward_value_images_offline': wandb.Image(images)}, step=i)
                images = agent.make_value_reward_visulization(variant, trajs[1])
                wandb_logger.log({'reward_value_images_online': wandb.Image(images)}, step=i)
        else:
            images = agent.make_value_reward_visulization(variant, trajs)
            wandb_logger.log({'reward_value_images': wandb.Image(images)}, step=i)
    else:
        if isinstance(trajs[0], list):
            images = agent.make_value_reward_visulization(variant, trajs[0][0])
            wandb_logger.log({'reward_value_images_offline_bridge': wandb.Image(images)}, step=i)
            images = agent.make_value_reward_visulization(variant, trajs[0][1])
            wandb_logger.log({'reward_value_images_offline_target': wandb.Image(images)}, step=i)
            images = agent.make_value_reward_visulization(variant, trajs[1])
            wandb_logger.log({'reward_value_images_online': wandb.Image(images)}, step=i)
        else:
            images = agent.make_value_reward_visulization(variant, trajs[0])
            wandb_logger.log({'reward_value_images_bridge': wandb.Image(images)}, step=i)
            images = agent.make_value_reward_visulization(variant, trajs[1])
            wandb_logger.log({'reward_value_images_target': wandb.Image(images)}, step=i)


def perform_control_eval(agent, eval_env, i, variant, wandb_logger):
    if variant.from_states:
        if hasattr(eval_env, 'enable_render'):
            eval_env.enable_render()
    eval_info = run_multiple_trajs(variant, agent,
                                   eval_env,
                                   num_trajs=variant.eval_episodes, deterministic=not variant.stochastic_evals)
    print('eval runs done.')
    if variant.from_states:
        if hasattr(eval_env, 'enable_render'):
            eval_env.disable_render()
    obs = eval_info.pop('obs')
    if 'pixels' in obs[0]:
        video = np.stack([ts['pixels'] for ts in obs]).squeeze()
        if len(video.shape) == 5:
            video = video[..., -1] # visualizing only the last frame of the stack when using framestacking
        video = video.transpose(0, 3, 1, 2)
        wandb_logger.log({'eval_video': wandb.Video(video[:, -3:], fps=8)}, step=i)

    if 'state' in obs[0] and variant.reward_type == 'dense':
        states = np.stack([ts['state'] for ts in obs])
        states_image = visualize_states_rewards(states, eval_info['rewards'], eval_env.target)
        wandb_logger.log({'state_traj_image': wandb.Image(states_image)}, step=i)

    for k, v in eval_info.items():
        if v.ndim == 0:
            wandb_logger.log({f'evaluation/{k}': v}, step=i)

    print('finished evals avg. return ', eval_info['return'])
    print('finished evals avg. length ', eval_info['length'])


def run_evals_only(variant, agent, eval_env, wandb_logger):
    i = 0
    while True:
        perform_control_eval(agent, eval_env, i, variant, wandb_logger)
        i += 1

def is_positive_sample(traj, i, variant, task_name):
    return i >= len(traj['observations']) - variant.num_final_reward_steps

def load_buffer(dataset_file, variant, task_aliasing_dict=None, multi_viewpoint=False, data_count_dict=None, split_pos_neg=False, num_traj_cutoff=None):
    print('loading buffer data from ', dataset_file)
    task_name = str.split(dataset_file, '/')[3 if variant.cond_interfing else -3]
    env_name = str.split(dataset_file, '/')[-3 if variant.cond_interfing else -4]
    if task_aliasing_dict and task_name in task_aliasing_dict:
        task_name = task_aliasing_dict[task_name]
    trajs = np.load(os.environ['DATA'] + dataset_file, allow_pickle=True)
    if data_count_dict is not None:
        if env_name not in data_count_dict:
            data_count_dict[env_name] = {}
        if task_name in data_count_dict[env_name]:
            data_count_dict[env_name][task_name] += len(trajs)
        else:
            data_count_dict[env_name][task_name] = len(trajs)
    if len(trajs) == 0:
        return 0, trajs

    pos_num_transitions = 0
    neg_num_transitions = 0
    num_transitions = 0

    # Count number of viewpoints
    if multi_viewpoint:
        viewpoints = trajs[0]['observations'][0].keys()
        viewpoints = [viewpoint for viewpoint in viewpoints if viewpoint.startswith('images')]
        num_viewpoints = len(viewpoints)
        print('num viewpoints', num_viewpoints)
    else:
        num_viewpoints = 1

    if num_traj_cutoff is not None:
        np.random.shuffle(trajs)
        trajs = trajs[:num_traj_cutoff]

    for traj in trajs:
        for i in range(len(traj['observations'])):
            if split_pos_neg:
                if is_positive_sample(traj, i, variant, task_name):
                    pos_num_transitions += num_viewpoints
                else:
                    neg_num_transitions += num_viewpoints
            else:
                num_transitions += num_viewpoints
        num_transitions += 1  # needed because of memory efficient replay buffer
        pos_num_transitions += 1  # needed because of memory efficient replay buffer
        neg_num_transitions += 1  # needed because of memory efficient replay buffer
        traj['task_description'] = task_name
    if split_pos_neg:
        return (pos_num_transitions, neg_num_transitions), trajs
    return num_transitions, trajs


def _reshape_image(obs):
    if len(obs.shape) == 1:
        obs = np.reshape(obs, (3, 128, 128))
        return np.transpose(obs, (1, 2, 0))
    elif len(obs.shape) == 3:
        return obs
    else:
        raise ValueError