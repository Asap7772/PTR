import glob
from jaxrl2.data.utils import include_tasks, exclude_tasks
import os

def get_openclose_all():
    tasks = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/*/*')

    include_list = ['open', 'close']
    exclude_list = ['book', 'pick_up_closest_rainbow_Allen_key_set', 'box']

    tasks = include_tasks(tasks, include_list)
    tasks = exclude_tasks(tasks, exclude_list)

    all_openclose_train = ['{}/train/out.npy'.format(task) for task in tasks]
    all_openclose_val = ['{}/val/out.npy'.format(task) for task in tasks]

    # print(all_openclose_train)
    task_names = [str.split(path,  '/')[-1] for path in tasks]
    # print('task_names', set(task_names))

    return all_openclose_train, all_openclose_val

def get_openclose_exclude_tk1():
    tasks = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/*/*')

    include_list = ['open', 'close']
    exclude_list = ['book', 'pick_up_closest_rainbow_Allen_key_set', 'box', 'toykitchen1']

    tasks = include_tasks(tasks, include_list)
    tasks = exclude_tasks(tasks, exclude_list)

    all_openclose_train = ['{}/train/out.npy'.format(task) for task in tasks]
    all_openclose_val = ['{}/val/out.npy'.format(task) for task in tasks]

    return all_openclose_train, all_openclose_val

def get_openclose_exclude_microwave():
    tasks = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/*/*')

    include_list = ['open', 'close']
    exclude_list = ['book', 'pick_up_closest_rainbow_Allen_key_set', 'box', 'microwave']

    tasks = include_tasks(tasks, include_list)
    tasks = exclude_tasks(tasks, exclude_list)

    all_openclose_train = ['{}/train/out.npy'.format(task) for task in tasks]
    all_openclose_val = ['{}/val/out.npy'.format(task) for task in tasks]

    return all_openclose_train, all_openclose_val

def get_tk1_targetdomain():
    tasks = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_data/toykitchen1/open_microwave']
    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_tk1_openmicro_failures():
    tasks = [os.environ['DATA'] + '/robonetv2/extracted_online_data/selected_online_negatives/open_microwave_online_negatives',
            os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/targetdomain_data/toykitchen1/open_microwave_negatives']

    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val