import os
import glob

target_task = 'PutBallintoBowl-v0'
target_task_interfering = 'task_1_0'

def get_multi_object_in_bowl_data():
    paths = os.listdir(os.environ['DATA'] + '/minibullet/multitask_pickplacedata_noise0.1')

    paths = filter_tasks(paths)

    train = [f'/minibullet/multitask_pickplacedata_noise0.1/{task}/train/out.npy' for task in paths]
    val = [f'/minibullet/multitask_pickplacedata_noise0.1/{task}/val/out.npy' for task in paths]
    return train, val

def get_multi_object_in_bowl_data_interfering():
    paths = os.listdir(os.environ['DATA'] + '/minibullet/0602_multitask_interfering')
    paths = filter_tasks(paths, target_task_interfering)

    train = [f'/minibullet/0602_multitask_interfering/{task}/PickPlaceInterferingDistractors-v0/train/out.npy' for task in paths]
    val = [f'/minibullet/multitask_pickplacedata_noise0.1/{task}/PickPlaceInterferingDistractors-v0/val/out.npy' for task in paths]
    
    for x in train:
        assert os.path.exists(os.environ['DATA'] + x), f'{os.environ["DATA"] + x} does not exist'
    
    return train, val


def filter_tasks(paths, target_task=target_task):
    new_paths = []
    for path in paths:
        if not target_task in path:
            new_paths.append(path)
    return new_paths


interfering_task=f'/minibullet/0602_multitask_interfering/{target_task_interfering}/PickPlaceInterferingDistractors-v0/train/out.npy'
interfering_task_val=f'/minibullet/0602_multitask_interfering/{target_task_interfering}/PickPlaceInterferingDistractors-v0/val/out.npy'

put_ball_in_bowl = '/minibullet/pickplacedata_noise0.1/PutBallintoBowl-v0/train/out.npy'
put_ball_in_bowl_val = '/minibullet/pickplacedata_noise0.1/PutBallintoBowl-v0/val/out.npy'


put_ball_in_bowl_delay1 = '/minibullet/pickplacedata_noise0.1_delay1/PutBallintoBowl-v0/train/out.npy'
put_ball_in_bowl_val_delay1 = '/minibullet/pickplacedata_noise0.1_delay1/PutBallintoBowl-v0/val/out.npy'

put_ball_in_bowl_neg = '/minibullet/pickplacedata_noise0.1_failonly250/PutBallintoBowl-v0/train/out.npy'
put_ball_in_bowl_neg_val = '/minibullet/pickplacedata_noise0.1_failonly250/PutBallintoBowl-v0/val/out.npy'

if __name__ == '__main__':
    os.environ['DATA'] = '/nfs/kun2/users/asap7772/bridge_data_exps/sim_data'
    print(get_multi_object_in_bowl_data_interfering())

