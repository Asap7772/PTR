from examples.configs.dataset_config_real import *
from examples.configs.toykitchen_pickplace_dataset import *
from examples.configs.get_door_openclose_data import *

import numpy as np
from gym.spaces import Box, Dict

class DummyEnv():
    def __init__(self, variant, num_tasks):
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

def get_train_target_tasks(variant):
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
        
    return train_tasks, eval_tasks, target_train_tasks, target_eval_tasks

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