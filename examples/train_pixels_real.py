#! /usr/bin/env python
from jaxrl2.utils.general_utils import AttrDict
import sys
import numpy as np

from collections import OrderedDict
import json
import datetime

from jaxrl2.agents import PixelBCLearner, PixelRewardLearner
from jaxrl2.wrappers.prev_action_wrapper import PrevActionStack
from jaxrl2.wrappers.state_wrapper import StateStack

from jaxrl2.agents.sac.sac_learner import SACLearner
from jaxrl2.agents.sarsa import PixelSARSALearner
from jaxrl2.agents.awbc.pixel_awbc_learner import PixelAWBCLearner
from jaxrl2.agents.cql_encodersep_parallel.pixel_cql_learner import PixelCQLLearnerEncoderSepParallel
from jaxrl2.agents.cql_encodersep_parallel_awbc.pixel_cql_learner import PixelAWBCLearnerEncoderSepParallel

from jaxrl2.wrappers.rescale_actions_wrapper import RescaleActions
from jaxrl2.wrappers.normalize_actions_wrapper import NormalizeActions
from examples.configs.dataset_config_real import *
from examples.configs.toykitchen_pickplace_dataset import *
from examples.configs.get_door_openclose_data import *

from gym.spaces import Dict
import sys

import gym
import numpy as np
from gym.spaces import Box

from jaxrl2.data import MemoryEfficientReplayBuffer, MemoryEfficientReplayBufferParallel, NaiveReplayBuffer, NaiveReplayBufferParallel 
from jaxrl2.data.dataset import MixingReplayBuffer, MixingReplayBufferParallel
from jaxrl2.data.replay_buffer import ReplayBuffer

from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
from jaxrl2.agents import PixelIQLLearner, PixelBCLearner
from jaxrl2.agents import IQLLearner

from examples.train_utils import offline_training_loop, trajwise_alternating_training_loop, load_buffer, run_evals_only, insert_data_real


from jaxrl2.wrappers import FrameStack
from jaxrl2.wrappers.reaching_reward_wrapper import ReachingReward
from jaxrl2.utils.general_utils import add_batch_dim
from jaxrl2.data.utils import get_task_id_mapping


TARGET_POINT = np.array([0.28425417, 0.04540814, 0.07545623])  # mean
# TARGET_POINT = np.array([0.23, 0., 0.1])


def main(variant):
    def wrap(env):
        assert not (variant.normalize_actions and variant.rescale_actions)
        if variant.reward_type == 'dense':
            env = ReachingReward(env, TARGET_POINT, variant.reward_type)
        if variant.add_prev_actions:
            if variant.frame_stack == 1:
                num_action_stack = 1
            else:
                num_action_stack = variant.frame_stack - 1
            env = PrevActionStack(env, num_action_stack)
        if variant.rescale_actions:
            print ('Rescaling actions in environment..............')
            env = RescaleActions(env)
        elif variant.normalize_actions:
            print ('Normalizing actions in environment.............')
            env = NormalizeActions(env)
        if variant.add_states:
            env = StateStack(env, variant.frame_stack)
        if not variant.from_states:
            env = FrameStack(env, variant.frame_stack)
        env = gym.wrappers.TimeLimit(env, variant.episode_timelimit)
        return env

    if variant.dataset == 'single_task':
        train_tasks = train_dataset_single_task
        eval_tasks = eval_dataset_single_task
    elif variant.dataset =='11tasks':
        print("using 11 tasks")
        train_tasks = train_dataset_11_task
        eval_tasks = eval_dataset_11_task
    elif variant.dataset == 'tk1_pickplace':
        train_tasks, eval_tasks = get_toykitchen1_pickplace()
    elif variant.dataset == 'tk2_pickplace':
        train_tasks, eval_tasks = get_toykitchen2_pickplace()
    elif variant.dataset == 'all_pickplace':
        train_tasks, eval_tasks = get_all_pickplace()
    elif variant.dataset == 'open_micro_single':
        train_tasks = train_dataset_single_task_openmicro
        eval_tasks = eval_dataset_single_task_openmicro
    elif variant.dataset == 'openclose_all' or variant.dataset == 'open_close':
        train_tasks, eval_tasks = get_openclose_all()
    elif variant.dataset == 'openclose_exclude_tk1':
        train_tasks, eval_tasks = get_openclose_exclude_tk1()
    elif variant.dataset == 'openclose_exclude_microwave':
        train_tasks, eval_tasks = get_openclose_exclude_microwave()
    elif variant.dataset == 'online_reaching_pixels':
        train_tasks = online_reaching_pixels
        eval_tasks = online_reaching_pixels_val
    elif variant.dataset == 'online_reaching_pixels_first100':
        train_tasks = online_reaching_pixels_first100
        eval_tasks = online_reaching_pixels_val_first100
    elif variant.dataset == 'toykitchen1_pickplace':
        train_tasks, eval_tasks = get_toykitchen1_pickplace()
    elif variant.dataset == 'toykitchen2_pickplace':
        train_tasks, eval_tasks = get_toykitchen2_pickplace()
    elif variant.dataset == 'all_pickplace':
        train_tasks, eval_tasks = get_all_pickplace()
    elif variant.dataset == 'all_pickplace_except_tk6':
        train_tasks, eval_tasks = get_all_pickplace_exclude_tk6()
    elif variant.dataset == 'toykitchen2_pickplace_simpler':
        train_tasks, eval_tasks = get_toykitchen2_pickplace_cardboardfence_reversible_simple()
    elif variant.dataset == 'toykitchen6_knife_in_pot':
        train_tasks, eval_tasks = get_toykitchen6_knife_in_pot()
    elif variant.dataset == 'toykitchen6_croissant_out_of_pot':
        train_tasks, eval_tasks = get_toykitchen6_croissant_out_of_pot()
    elif variant.dataset == 'toykitchen6_pear_from_plate':
        train_tasks, eval_tasks = get_toyktichen6_pear_from_plate()
    elif variant.dataset == 'toykitchen6_sweet_potato_on_plate':
        train_tasks, eval_tasks = get_toykitchen6_put_sweet_potato_on_plate()
    elif variant.dataset == 'toykitchen6_sweet_potato_in_bowl':
        train_tasks, eval_tasks = get_toykitchen6_put_sweet_potato_in_bowl()
    elif variant.dataset == 'toykitchen6_lime_in_pan_sink':
        train_tasks, eval_tasks = get_toyktichen6_put_lime_in_pan_sink()
    elif variant.dataset == 'toykitchen6_drumstick_on_plate':
        train_tasks, eval_tasks = get_toykitchen6_put_drumstick_on_plate()
    elif variant.dataset == 'toykitchen6_cucumber_in_pot':
        train_tasks, eval_tasks = get_toykitchen6_cucumber_in_orange_pot()
    elif variant.dataset == 'toykitchen6_carrot_in_pan':
        train_tasks, eval_tasks = get_toykitchen6_carrot_in_pan()
    elif variant.dataset == 'r3m_cucumber':
        train_tasks, eval_tasks = get_tk6_r3m_cucumber()
    elif variant.dataset == 'r3m_croissant':
        train_tasks, eval_tasks = get_tk6_r3m_croissant()
    elif variant.dataset == 'r3m_knife':
        train_tasks, eval_tasks = get_tk6_r3m_knife()
    elif variant.dataset == 'r3m_sweet_potato_plate':
        train_tasks, eval_tasks = get_tk6_r3m_sweet_potato_plate()
    elif variant.dataset == 'debug':
        num_debug_tasks = 3
        train_tasks, eval_tasks = get_all_pickplace_exclude_tk6()
        train_tasks = train_tasks[:num_debug_tasks]
        eval_tasks = eval_tasks[:num_debug_tasks]
    else:
        raise ValueError('dataset not found! ' + variant.dataset)

    if variant.target_dataset != '':
        if variant.target_dataset == 'toykitchen2_pickplace_cardboardfence_reversible':
            target_train_tasks, target_eval_tasks = get_toykitchen2_pickplace_cardboardfence_reversible()
            # target_train_tasks, target_eval_tasks = get_toykitchen2_pickplace_cardboardfence_reversible_simple()
        elif variant.target_dataset == 'toykitchen2_pickplace_simpler':
            target_train_tasks, target_eval_tasks = get_toykitchen2_pickplace_cardboardfence_reversible_simple()
        elif variant.target_dataset == 'toykitchen6_pickplace_reversible':
            target_train_tasks, target_eval_tasks = get_toykitchen6_pickplace_reversible()
        elif variant.target_dataset == 'toykitchen6_target_domain':
            target_train_tasks, target_eval_tasks = get_toykitchen6_target_domain()
        elif variant.target_dataset == 'toykitchen6_new_target_domain':
            target_train_tasks, target_eval_tasks = get_toykitchen6_new_target_domain()
        elif variant.target_dataset == 'toykitchen6_target_domain_two_tasks':
            target_train_tasks, target_eval_tasks = get_toykitchen6_new_target_domain_2_tasks()
        elif variant.target_dataset == 'toykitchen6_target_domain_five_tasks':
            target_train_tasks, target_eval_tasks = get_toykitchen6_new_target_domain_5_tasks()
        elif variant.target_dataset == 'toykitchen6_knife_in_pot':
            target_train_tasks, target_eval_tasks = get_toykitchen6_knife_in_pot()
        elif variant.target_dataset == 'toykitchen6_croissant_out_of_pot':
            target_train_tasks, target_eval_tasks = get_toykitchen6_croissant_out_of_pot()
        elif variant.target_dataset == 'toykitchen6_pear_from_plate':
            target_train_tasks, target_eval_tasks = get_toyktichen6_pear_from_plate()
        elif variant.target_dataset == 'toykitchen6_sweet_potato_on_plate':
            target_train_tasks, target_eval_tasks = get_toykitchen6_put_sweet_potato_on_plate()
        elif variant.target_dataset == 'toykitchen6_sweet_potato_in_bowl':
            target_train_tasks, target_eval_tasks = get_toykitchen6_put_sweet_potato_in_bowl()
        elif variant.target_dataset == 'toykitchen6_lime_in_pan_sink':
            target_train_tasks, target_eval_tasks = get_toyktichen6_put_lime_in_pan_sink()
        elif variant.target_dataset == 'toykitchen6_drumstick_on_plate':
            target_train_tasks, target_eval_tasks = get_toykitchen6_put_drumstick_on_plate()
        elif variant.target_dataset == 'toykitchen6_cucumber_in_pot':
            target_train_tasks, target_eval_tasks = get_toykitchen6_cucumber_in_orange_pot()
        elif variant.target_dataset == 'toykitchen6_carrot_in_pan':
            target_train_tasks, target_eval_tasks = get_toykitchen6_carrot_in_pan()
        elif variant.target_dataset == 'toykitchen6_big_corn_in_big_pot':
            target_train_tasks, target_eval_tasks = get_toykitchen6_big_corn_in_big_pot()
        elif variant.target_dataset == 'toykitchen1_pickplace_cardboardfence_reversible':
            target_train_tasks, target_eval_tasks = get_toykitchen1_pickplace_cardboardfence_reversible()
        elif variant.target_dataset == 'toykitchen2_sushi_targetdomain':
            target_train_tasks, target_eval_tasks = get_toykitchen2_sushi_targetdomain()
        elif variant.target_dataset == 'tk1_target_openmicrowave':
            target_train_tasks, target_eval_tasks = get_tk1_targetdomain()
        elif variant.target_dataset == 'r3m_cucumber':
            target_train_tasks, target_eval_tasks = get_tk6_r3m_cucumber()
        elif variant.target_dataset == 'r3m_croissant':
            target_train_tasks, target_eval_tasks = get_tk6_r3m_croissant()
        elif variant.target_dataset == 'r3m_knife':
            target_train_tasks, target_eval_tasks = get_tk6_r3m_knife()
        elif variant.target_dataset == 'r3m_sweet_potato_plate':
            target_train_tasks, target_eval_tasks = get_tk6_r3m_sweet_potato_plate()
        elif variant.target_dataset == 'cucumber_rotated_elevated':
            target_train_tasks, target_eval_tasks = get_toykitchen6_cucumber_rotated_elevated()
        elif variant.target_dataset == 'cucumber_elevated':
            target_train_tasks, target_eval_tasks = get_toykitchen6_cucumber_elevated()
        elif variant.target_dataset == 'croissant_rotated':
            target_train_tasks, target_eval_tasks = get_toykitchen6_croissant_rotated()
        elif variant.target_dataset == 'croissant_elevated':
            target_train_tasks, target_eval_tasks = get_toykitchen6_croissant_elevated()
        elif variant.target_dataset == 'debug':
            num_debug_tasks=1
            target_train_tasks, target_eval_tasks = get_toykitchen2_pickplace_cardboardfence_reversible_simple()
            target_train_tasks = target_train_tasks[:num_debug_tasks]
            target_eval_tasks = target_eval_tasks[:num_debug_tasks]
        else:
            raise ValueError('target dataset not found! ' + variant.target_dataset)
    else:
        target_train_tasks = []
        target_eval_tasks = []

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

    if not variant.eval_only and variant.offline_only:
        class DummyEnv():
            def __init__(self):
                super().__init__()
                obs_dict = dict()
                if not variant.from_states:
                    if variant['encoder_type'] == 'identity':
                        print("using identity encoder with R3M")
                        obs_dict['pixels'] = Box(low=-100000, high=100000, shape=(2048,), dtype=np.float32)
                    else:
                        print("using image encoder for Pixels")
                        obs_dict['pixels'] = Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
                if variant.add_states:
                    obs_dict['state'] = Box(low=-100000, high=100000, shape=(7,), dtype=np.float32)
                if num_tasks > 1:
                    obs_dict['task_id'] = Box(low=0, high=1, shape=(num_tasks,), dtype=np.float32)
                self.observation_space = Dict(obs_dict)
                self.spec = None
                self.action_space = Box(
                    np.asarray([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.]),
                    np.asarray([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0]),
                    dtype=np.float32)

            def seed(self, seed):
                pass

        env = DummyEnv()
    else:
        from jaxrl2.extra_envs.widowx_real_env import get_env_params
        env_params = get_env_params(variant)
        env_params['add_task_id'] = num_tasks > 1
        if variant.reward_type == 'classifier':
            from jaxrl2.extra_envs.widowx_real_env import BridgeDataJaxRLWidowXRewardAdapter, BridgeDataJaxRLVRWidowXReward
            if variant.algorithm == 'vr' or variant.algorithm.startswith('dagger'):
                env = BridgeDataJaxRLVRWidowXReward(env_params, num_tasks=num_tasks, task_id=variant.eval_task_id,
                                                    all_tasks=variant.all_tasks, task_id_mapping=task_id_mapping)
            else:
                env = BridgeDataJaxRLWidowXRewardAdapter(env_params, num_tasks=num_tasks, task_id=variant.eval_task_id,
                                                         all_tasks=variant.all_tasks, task_id_mapping=task_id_mapping)
                if len(variant.all_tasks) != 0:
                    assert variant.eval_task_id == -1  # make sure to allow automatic task selection through setting to -1
        elif not variant.from_states and variant.reward_type == 'dense':
            from jaxrl2.extra_envs.widowx_real_env import ImageReachingJaxRLWidowXEnv
            env = ImageReachingJaxRLWidowXEnv(env_params, num_tasks=num_tasks)
        else:
            from jaxrl2.extra_envs.widowx_real_env import JaxRLWidowXEnv
            env = JaxRLWidowXEnv(env_params, num_tasks=num_tasks)
    env = wrap(env)
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

    if variant.reward_type == 'classifier':
        reward_kwargs = variant.variant_reward['train_kwargs']
        sample_obs_rew = sample_obs.copy()
        if 'state' in sample_obs_rew:
            sample_obs_rew.pop('state')
        reward_function = PixelRewardLearner(variant.seed, sample_obs_rew, sample_action, **reward_kwargs)
        print('loading checkpoint reward classifier ...')
        reward_function.restore_checkpoint(variant.restore_reward_path)
        if not variant.offline_only or variant.eval_only:
            env.set_reward_function(reward_function)
    else:
        reward_function = None

    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps

    expname = create_exp_name(variant.prefix, seed=variant.seed)
    outputdir = os.environ['EXP'] + '/jaxrl/' + expname
    variant.outputdir = outputdir
    print('writing to output dir ', outputdir)

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=outputdir, group_name=group_name)

    if variant.from_states:
        if variant.algorithm == 'iql':
            agent = IQLLearner(variant.seed, sample_obs, sample_action
                               , **kwargs)
        elif variant.algorithm == 'sac':
            agent = SACLearner(variant.seed, env.observation_space,
                               env.action_space, **kwargs)
    else:
        if variant.algorithm == 'iql':
            agent = PixelIQLLearner(variant.seed, sample_obs,
                                sample_action, **kwargs)
        elif variant.algorithm == 'bc':
            agent = PixelBCLearner(variant.seed, sample_obs,
                                    sample_action, **kwargs)
        elif variant.algorithm == 'awbc':
            agent = PixelAWBCLearnerEncoderSepParallel(variant.seed, sample_obs, 
                                     sample_action, **kwargs)
        elif variant.algorithm == 'sarsa':
            agent = PixelSARSALearner(variant.seed, sample_obs, 
                                    sample_action, **kwargs)
        elif variant.algorithm == 'cql_encodersep_parallel':
            agent = PixelCQLLearnerEncoderSepParallel(variant.seed, sample_obs, sample_action, **kwargs)
        else:
            assert False, 'unknown algorithm'

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

    if variant.eval_only:
        run_evals_only(variant, agent, eval_env, wandb_logger)
        print('evals done exiting.')
        sys.exit()

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

    if not variant.online_from_scratch:
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
        if variant.offline_only or variant.onthespot_offline_train:
            offline_training_loop(variant, agent, eval_env, train_replay_buffer, eval_replay_buffer, wandb_logger,
                                  perform_control_evals=False, task_id_mapping=task_id_mapping)

        online_replay_buffer = replay_buffer_class(env.observation_space, env.action_space, int(5e5))
        replay_buffer = mixing_buff_class([train_replay_buffer, online_replay_buffer], variant.online_mixing_ratio)
        print('buffer len', replay_buffer.length())
    else:
        replay_buffer = online_replay_buffer = replay_buffer_class(env.observation_space, env.action_space, int(5e5))
        save_jsons(outputdir, task_id_mapping, variant)

    if variant.restore_path != '':
        restore_folder = '/'.join(str.split(variant.restore_path, '/')[:-1])
        if 'replaybuffer.npy' in os.listdir(restore_folder):
            online_replay_buffer.restore(restore_folder + '/replaybuffer.npy')
        print("restored replay buffer")

    replay_buffer.seed(variant.seed)

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    from jaxrl2.data.raw_saver import RawSaverJaxRL
    saver = RawSaverJaxRL(os.environ['DATA'] + '/robonetv2/online_datacollection/{}/{}'.format(variant.prefix, now),
                          env.unnormalize_actions)

    trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer,
                                           wandb_logger, saver=saver, perform_control_evals=variant.perform_control_evals if 'perform_control_evals' in variant else True)


def make_buffer_and_insert(env, replay_buffer_class, task_id_mapping, tasks, variant, 
                           split_pos_neg=False, reward_function=None,
                           num_traj_cuttoff=-1):
    pos_buffer_size = neg_buffer_size = buffer_size = 0
    all_trajs = []
    data_count_dict = {}
    for dataset_file in tasks:
        task_size, trajs = load_buffer(dataset_file, variant, ALIASING_DICT, multi_viewpoint=variant.multi_viewpoint,
                                       data_count_dict=data_count_dict, split_pos_neg=split_pos_neg,
                                       num_traj_cutoff=num_traj_cuttoff)
        if split_pos_neg:
            pos_size, neg_size = task_size
            pos_buffer_size += pos_size
            neg_buffer_size += neg_size
        else:
            buffer_size += task_size
        all_trajs.append(trajs)
    print('size ', buffer_size)
    if split_pos_neg:
        pos_buffer = replay_buffer_class(env.observation_space, env.action_space, pos_buffer_size)
        neg_buffer = replay_buffer_class(env.observation_space, env.action_space, neg_buffer_size)
        buffer = MixingReplayBuffer([pos_buffer, neg_buffer], 0.5)
    else:
        buffer = replay_buffer_class(env.observation_space, env.action_space, buffer_size)
    print('inserting data...')
    for trajs in all_trajs:
        if variant.multi_viewpoint:
            [insert_data_real(variant, buffer, trajs, variant.reward_type, task_id_mapping, env=env,
                              image_key='images' + str(i), target_point=TARGET_POINT, split_pos_neg=split_pos_neg, reward_function=reward_function) for i in range(3)]
        else:
            insert_data_real(variant, buffer, trajs, variant.reward_type, task_id_mapping, env=env,
                             image_key='images0', target_point=TARGET_POINT, split_pos_neg=split_pos_neg, reward_function=reward_function)
    return buffer, data_count_dict


def save_jsons(outputdir, task_id_mapping, variant, data_count_dict=None, data_count_dict_val=None, data_count_dict_target=None):
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('saving config to ', outputdir)
    def save_ordered_dict(output_dir, name, dict):
        with open(os.path.join(output_dir, "{}.json".format(name)), 'w') as f:
            variant = OrderedDict(sorted(OrderedDict(dict).items(), key=lambda x: x))
            json.dump(variant, f, indent=4)
    save_ordered_dict(outputdir, 'task_index', task_id_mapping)
    save_ordered_dict(outputdir, 'data_count_dict', data_count_dict)
    save_ordered_dict(outputdir, 'data_count_dict_val', data_count_dict_val)
    if data_count_dict_target is not None:
        save_ordered_dict(outputdir, 'data_count_dict_target', data_count_dict_target)
    print('data count dict train', data_count_dict)
    print('data count dict val', data_count_dict_val)
    save_ordered_dict(outputdir, 'config', variant)

def save_action_stats(action_stats, path):
    np.save(path + '/action_stats.npy', [action_stats])

def load_action_stats(path):
    return np.load(path + '/action_stats.npy', allow_pickle=True)