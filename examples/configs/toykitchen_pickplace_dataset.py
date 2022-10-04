import glob
import os
from jaxrl2.data.utils import include_tasks, exclude_tasks

def get_tk6_r3m_cucumber():
    tasks = ['put_cucumber_in_orange_pot']
    train = [os.environ['DATA'] + f'robonetv2/toykitchen_numpy_shifted/normalized_r3m_feat/{task}/train/out.npy' for task in tasks]
    val = [os.environ['DATA'] + f'robonetv2/toykitchen_numpy_shifted/normalized_r3m_feat/{task}/val/out.npy' for task in tasks]
    return train, val

def get_tk6_r3m_croissant():
    tasks = ['take_croissant_out_of_pot']
    train = [os.environ['DATA'] + f'robonetv2/toykitchen_numpy_shifted/normalized_r3m_feat/{task}/train/out.npy' for task in tasks]
    val = [os.environ['DATA'] + f'robonetv2/toykitchen_numpy_shifted/normalized_r3m_feat/{task}/val/out.npy' for task in tasks]
    return train, val

def get_tk6_r3m_knife():
    tasks = ['put_knife_into_pot']
    train = [os.environ['DATA'] + f'robonetv2/toykitchen_numpy_shifted/normalized_r3m_feat/{task}/train/out.npy' for task in tasks]
    val = [os.environ['DATA'] + f'robonetv2/toykitchen_numpy_shifted/normalized_r3m_feat/{task}/val/out.npy' for task in tasks]
    return train, val

def get_tk6_r3m_sweet_potato_plate():
    tasks = ['put_sweet_potato_on_plate']
    train = [os.environ['DATA'] + f'robonetv2/toykitchen_numpy_shifted/normalized_r3m_feat/{task}/train/out.npy' for task in tasks]
    val = [os.environ['DATA'] + f'robonetv2/toykitchen_numpy_shifted/normalized_r3m_feat/{task}/val/out.npy' for task in tasks]
    return train, val

def get_toykitchen2_pickplace():
    tasks = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen2/*')
    tasks += glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen2_room8052/*')
    exclude_list = ['zip', 'test', 'box', 'fold_cloth_in_half', 'put_cap_on_container', 'open_book', 'basil_bottle', 'flip',
                    'topple', 'open', 'close', 'light_switch', 'upright', 'pour','drying_rack', 'faucet', 'turn']
    tasks = exclude_tasks(tasks, exclude_list)
    task_names = [str.split(path,  '/')[-1] for path in tasks]
    print(task_names)

    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen2_pickplace_cardboardfence_reversible():
    tasks = ['put_lid_on_pot_cardboardfence',
             'take_lid_off_pot_cardboardfence',
             'put_bowl_on_plate_cardboard_fence',
             'take_bowl_off_plate_cardboard_fence',
             'put_sushi_in_pot_cardboard_fence',
             'take_sushi_out_of_pot_cardboard_fence',
             'put_carrot_in_pot_cardboard_fence',
             'take_carrot_out_of_pot_cardboard_fence',
             ]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen2/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen2/{}/val/out.npy'.format(task) for task in tasks]
    return train, val


def get_toykitchen2_sushi_targetdomain():
    tasks = ['put_sushi_in_pot_cardboard_fence', 'take_sushi_out_of_pot_cardboard_fence',]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain/toykitchen2/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain/toykitchen2/{}/val/out.npy'.format(task) for task in tasks]
    return train, val




def get_toykitchen2_pickplace_cardboardfence_reversible_simple():
    tasks = ['put_lid_on_pot_cardboardfence',
            #  'take_lid_off_pot_cardboardfence',
             'put_bowl_on_plate_cardboard_fence',
            #  'take_bowl_off_plate_cardboard_fence',
             'put_sushi_in_pot_cardboard_fence',
            #  'take_sushi_out_of_pot_cardboard_fence',
             'put_carrot_in_pot_cardboard_fence',
            #  'take_carrot_out_of_pot_cardboard_fence',
             ]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen2/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen2/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_pickplace_reversible():
    tasks = ['put_corn_in_bowl_sink',
            'take_corn_out_of_bowl_sink',
            'put_spoon_in_bowl_sink',
            'take_spoon_out_of_bowl_sink',
             'put_cup_on_plate',
             'take_cup_off_plate'
             ]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_target_domain():
    tasks = ['put_knife_into_pot',
             'take_croissant_out_of_pot',]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-15/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-15/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_new_target_domain():
    tasks = ['put_knife_into_pot',
             'take_croissant_out_of_pot',
             'take_pear_from_plate',
             'put_sweet_potato_on_plate',
             'put_lime_in_pan_sink',
             'put_sweet_potato_in_bowl',
             'put_drumstick_on_plate',
             'put_cucumber_in_orange_pot',
             'put_carrot_in_pan',
             'put_big_corn_in_big_pot']
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val


def get_toykitchen6_knife_in_pot():
    tasks = ['put_knife_into_pot',]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_croissant_out_of_pot():
    tasks = ['take_croissant_out_of_pot',]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toyktichen6_pear_from_plate():
    tasks = ['take_pear_from_plate',]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_put_sweet_potato_on_plate():
    tasks = ['put_sweet_potato_on_plate',]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_put_sweet_potato_in_bowl():
    tasks = ['put_sweet_potato_in_bowl',]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toyktichen6_put_lime_in_pan_sink():
    tasks = ['put_lime_in_pan_sink',]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_put_drumstick_on_plate():
    tasks = ['put_drumstick_on_plate',]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_cucumber_in_orange_pot():
    tasks = ['put_cucumber_in_orange_pot',]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_carrot_in_pan():
    tasks = ['put_carrot_in_pan',]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_big_corn_in_big_pot():
    tasks = ['put_big_corn_in_big_pot',]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val



def get_toykitchen6_new_target_domain_5_tasks():
    tasks = ['put_knife_into_pot',
             'take_croissant_out_of_pot',
             'take_pear_from_plate',
             'put_sweet_potato_on_plate',
             'put_lime_in_pan_sink'
    ]
            #  'put_sweet_potato_in_bowl',
            #  'put_drumstick_on_plate',
            #  'put_cucumber_in_orange_pot',
            #  'put_carrot_in_pan',
            #  'put_big_corn_in_big_pot']
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_new_target_domain_2_tasks():
    tasks = [
        # 'put_knife_into_pot',
            #  'take_croissant_out_of_pot',
            #  'take_pear_from_plate',
            #  'put_sweet_potato_on_plate',
             'put_lime_in_pan_sink',
            #  'put_sweet_potato_in_bowl',
             'put_drumstick_on_plate',
    ]
            #  'put_cucumber_in_orange_pot',
            #  'put_carrot_in_pan',
            #  'put_big_corn_in_big_pot']
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val



def get_toykitchen1_pickplace():
    tasks = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/*')
    exclude_list = ['test', 'box', 'basket', 'knob', 'open', 'close', 'flip', 'lever']
    tasks = exclude_tasks(tasks, exclude_list)
    task_names = [str.split(path,  '/')[-1] for path in tasks]
    for name in task_names:
        print(name)
    print(len(task_names))

    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen1_pickplace_cardboardfence_reversible():
    tasks = [
        'put_broccoli_in_pan_cardboardfence',
        'put_carrot_on_plate_cardboardfence',
        'take_broccoli_out_of_pan_cardboardfence',
        'take_carrot_off_plate_cardboardfence'
    ]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/{}/val/out.npy'.format(task) for task in tasks]
    return train, val


def get_all_pickplace():
    tasks = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/*/*')
    exclude_list = ['zip', 'test', 'box', 'fold_cloth_in_half', 'put_cap_on_container', 'open_book', 'basil_bottle',
                    'test', 'box', 'basket', 'knob', 'open', 'close', 'flip', 'lever', 'topple', 'pour', 'drying_rack'
                    ]
    tasks = exclude_tasks(tasks, exclude_list)
    task_names = [str.split(path,  '/')[-2:] for path in tasks]
    for name in task_names:
        print(name)
    print(len(task_names))

    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_all_pickplace_exclude_tk6():
    tasks = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/*/*')
    exclude_list = ['zip', 'test', 'box', 'fold_cloth_in_half', 'put_cap_on_container', 'open_book', 'basil_bottle',
                    'test', 'box', 'basket', 'knob', 'open', 'close', 'flip', 'lever', 'topple', 'pour', 'drying_rack',
                    'tool_chest', 'laundry_machine']
    exclude_list += ['toykitchen6']

    tasks = exclude_tasks(tasks, exclude_list)
    task_names = [str.split(path,  '/')[-2:] for path in tasks]
    for name in task_names:
        print(name)
    print(len(task_names))

    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_cucumber_elevated():
    tasks = ['put_cucumber_in_orange_pot',]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-15/toykitchen6_elevated/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-15/toykitchen6_elevated/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_cucumber_rotated_elevated():
    tasks = ['put_cucumber_in_orange_pot',]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-15/toykitchen6_rotated_elevated/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-15/toykitchen6_rotated_elevated/{}/val/out.npy'.format(task) for task in tasks]
    return train, val


def get_toykitchen6_croissant_elevated():
    tasks = ['take_croissant_out_of_pot',]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-15/toykitchen6_elevated/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-15/toykitchen6_elevated/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_croissant_rotated():
    tasks = ['take_croissant_out_of_pot',]
    train = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-15/toykitchen6_rotated/{}/train/out.npy'.format(task) for task in tasks]
    val = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-15/toykitchen6_rotated/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

if __name__ == '__main__':
    print(get_toykitchen6_cucumber_elevated()[0])
    # get_toykitchen1_pickplace()
    # get_all_pickplace()
    # get_toykitchen2_pickplace()
    # get_toykitchen2_pickplace_cardboardfence_reversible()