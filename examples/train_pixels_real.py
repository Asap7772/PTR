#! /usr/bin/env python
import numpy as np
import sys
import gym

from jaxrl2.data import  NaiveReplayBuffer, NaiveReplayBufferParallel 
from jaxrl2.data.dataset import MixingReplayBuffer, MixingReplayBufferParallel
from jaxrl2.data.replay_buffer import ReplayBuffer
from jaxrl2.data.utils import get_task_id_mapping

from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
from jaxrl2.utils.agent_util import get_algo
from jaxrl2.utils.dataset_util import *
from jaxrl2.utils.env_util import wrap
from jaxrl2.utils.general_utils import add_batch_dim

from examples.train_utils import offline_training_loop
from examples.configs.dataset_config_real import ALIASING_DICT

def main(variant):
    train_tasks, eval_tasks, target_train_tasks, target_eval_tasks = get_train_target_tasks(variant)
    
    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps

    if 'hidden_dims' in variant['train_kwargs']:
        variant['train_kwargs']['hidden_dims'] = tuple(variant['train_kwargs']['hidden_dims'])
    if 'variant_reward' in variant:
        if 'hidden_dims' in variant.variant_reward['train_kwargs']:
            variant.variant_reward['train_kwargs']['hidden_dims'] = tuple(
                variant.variant_reward['train_kwargs']['hidden_dims'])

    if variant.num_eval_tasks == -1:
        task_id_mapping = get_task_id_mapping(train_tasks + target_train_tasks, ALIASING_DICT)
        print('task_id_mapping:', task_id_mapping)
        num_tasks = len(task_id_mapping.keys())
        variant.num_tasks = num_tasks
        variant.task_id_mapping = task_id_mapping
        print('using {} tasks'.format(num_tasks))
    else:
        num_tasks = variant.num_eval_tasks
        task_id_mapping = None

    env = DummyEnv(variant, num_tasks)
    env = wrap(env, variant)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(variant.seed)
    if variant.from_states:
        env.disable_render()
    if variant.reward_type == 'dense':
        assert not variant.add_states
    eval_env = env

    sample_obs = add_batch_dim(env.observation_space.sample())
    sample_action = add_batch_dim(env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    reward_function = None
    
    agent = get_algo(variant, sample_obs, sample_action, **kwargs)

    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps

    expname = create_exp_name(variant.prefix, seed=variant.seed)
    outputdir = os.environ['EXP'] + '/jaxrl/' + expname
    variant.outputdir = outputdir
    print('writing to output dir ', outputdir)

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=outputdir, group_name=group_name)

    if variant.restore_path != '':
        print('loading checkpoint...')
        agent.restore_checkpoint(variant.restore_path)
        if 'parallel' in variant.algorithm:
            agent.replicate()
        if variant.normalize_actions:
            action_stats = load_action_stats('/'.join(variant.restore_path.split('/')[:-1]))[0]
            if not variant.offline_only or variant.eval_only:
                env.set_action_stats(action_stats)
                eval_env.set_action_stats(action_stats)
            print('restored action stats.')
    else:
        action_stats = {'mean': np.zeros_like(env.action_space.low), 'std': np.ones_like(env.action_space.low)}
        if not variant.offline_only or variant.eval_only:
            env.set_action_stats(action_stats)
            eval_env.set_action_stats(action_stats)

    if variant.from_states:
        replay_buffer_class = ReplayBuffer
    elif variant.algorithm in ['cql_parallel', 'cql_encodersep_parallel', 'awbc']:
        if variant.target_dataset != '':
            replay_buffer_class = NaiveReplayBuffer
        else:
            replay_buffer_class = NaiveReplayBufferParallel
    else:
        replay_buffer_class = NaiveReplayBuffer
    if variant.algorithm in ['cql_encodersep_parallel', 'cql_parallel', 'awbc']:
        mixing_buff_class = MixingReplayBufferParallel
    else:
        mixing_buff_class = MixingReplayBuffer

    
    print('making train buffer:')
    split_pos_neg = variant.split_pos_neg if 'split_pos_neg' in variant else False
    train_replay_buffer, data_count_dict = make_buffer_and_insert(env, replay_buffer_class, task_id_mapping, train_tasks, variant, split_pos_neg=split_pos_neg, reward_function=reward_function)
    print('making val buffer:')
    eval_replay_buffer, data_count_dict_val = make_buffer_and_insert(env, replay_buffer_class, task_id_mapping, eval_tasks, variant, split_pos_neg=False, reward_function=reward_function)

    if 'target_dataset' in variant and variant.target_dataset != '':
        target_replay_buffer_train, data_count_dict_target = make_buffer_and_insert(
            env, replay_buffer_class, task_id_mapping, target_train_tasks, variant,
            split_pos_neg=split_pos_neg, reward_function=reward_function,
            num_traj_cuttoff=variant.num_target_traj)
        train_replay_buffer = mixing_buff_class([train_replay_buffer, target_replay_buffer_train], variant.target_mixing_ratio)
        target_replay_buffer_eval, _ = make_buffer_and_insert(
            env, replay_buffer_class, task_id_mapping, target_eval_tasks, variant, 
            split_pos_neg=split_pos_neg, reward_function=reward_function,
            num_traj_cuttoff=variant.num_target_traj)
        eval_replay_buffer = mixing_buff_class([eval_replay_buffer, target_replay_buffer_eval], variant.target_mixing_ratio)
    else:
        data_count_dict_target = None

    action_stats = train_replay_buffer.compute_action_stats()
    print('dataset action stats ', action_stats)
    if variant.normalize_actions:
        train_replay_buffer.normalize_actions(action_stats)
        eval_replay_buffer.normalize_actions(action_stats)
        print('dataset action stats after norm', train_replay_buffer.compute_action_stats())
        env.set_action_stats(action_stats)
        eval_env.set_action_stats(action_stats)
    save_action_stats(action_stats, variant.outputdir)

    save_jsons(outputdir, task_id_mapping, variant, data_count_dict, data_count_dict_val, data_count_dict_target)

    train_replay_buffer.seed(variant.seed)
    eval_replay_buffer.seed(variant.seed)
    
    offline_training_loop(
        variant, 
        agent, 
        eval_env, 
        train_replay_buffer, 
        eval_replay_buffer, 
        wandb_logger,
        perform_control_evals=False, 
        task_id_mapping=task_id_mapping
    )