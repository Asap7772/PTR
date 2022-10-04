import copy
from typing import Optional, Union

import gym.spaces
import numpy as np

from jaxrl2.data.dataset import Dataset, DatasetDict
import collections
from typing import Iterable, Optional
import gym
import jax


def _init_replay_dict(obs_space: gym.Space,
                      capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(dataset_dict: DatasetDict, data_dict: DatasetDict,
                        insert_index: int):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys(), 'dataset_dict.keys() {} != data_dict.keys() {}'.format(dataset_dict.keys(), data_dict.keys())
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


def _slice_recursively(source_dict: DatasetDict, slice_index: int):
    target_dict = {}
    for k in source_dict.keys():
        if isinstance(source_dict[k], np.ndarray):
            target_dict[k] = source_dict[k][:slice_index]
        elif isinstance(source_dict, dict):
            target_dict[k] = _slice_recursively(source_dict[k], slice_index)
        else:
            raise TypeError()
    return target_dict

def _insert_snapshot_recursively(source_dict: DatasetDict, target_dict: DatasetDict):
    for k in source_dict.keys():
        if isinstance(source_dict[k], np.ndarray):
            index = source_dict[k].shape[0]
            target_dict[k][:index] = source_dict[k]
        elif isinstance(source_dict, dict):
            _insert_snapshot_recursively(source_dict[k], target_dict[k])
        else:
            raise TypeError()


class ReplayBuffer(Dataset):

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 capacity: int,
                 next_observation_space: Optional[gym.Space] = None):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space,
                                                  capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape),
                             dtype=action_space.dtype),
            next_actions=np.empty((capacity, *action_space.shape),
                             dtype=action_space.dtype),
            rewards=np.empty((capacity, ), dtype=np.float32),
            masks=np.empty((capacity, ), dtype=np.float32),
            dones=np.empty((capacity, ), dtype=bool),
            trajectory_id=np.empty((capacity, ), dtype=np.int)
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0
        self._traj_counter = 1   # start at 1 to avoid selecting the whole buffer when sampling whole trajs

    def __len__(self) -> int:
        return self._size

    def length(self) -> int:
        return self._size

    def increment_traj_counter(self):
        self._traj_counter += 1

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(self,
                     batch_size: int,
                     keys: Optional[Iterable[str]] = None,
                     indx: Optional[np.ndarray] = None,
                     queue_size: int = 2):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size, keys, indx)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def get_random_trajs(self, batch_size):
        observations_list = []
        actions_list = []
        next_actions_list = []
        rewards_list = []
        masks_list = []
        for i in range(batch_size):
            valid_traj_found = False
            i_sample = 0
            while not valid_traj_found:
                if i_sample > 5:
                    raise ValueError('could not sample trajectory with more than 10 steps!')
                i_sample += 1
                if hasattr(self.np_random, 'integers'):
                    indx = self.np_random.integers(len(self), size=1)
                else:
                    indx = self.np_random.randint(len(self), size=1)
                start_ind, end_ind, consecutive = self.get_traj_start_end_indices(indx)
                if consecutive:
                    valid_traj_found = True
                if (end_ind - start_ind) < 10:
                    valid_traj_found = False

            all_observation_keys = {}
            for key in self.dataset_dict['observations']:
                all_observation_keys[key] = self.dataset_dict['observations'][key][start_ind: end_ind + 1]

            observations_list.append(all_observation_keys)
            actions_list.append(self.dataset_dict['actions'][start_ind: end_ind + 1].squeeze())
            next_actions_list.append(self.dataset_dict['actions'][start_ind: end_ind + 1].squeeze())
            rewards_list.append(self.dataset_dict['rewards'][start_ind: end_ind + 1].squeeze())
            masks_list.append(self.dataset_dict['masks'][start_ind: end_ind + 1].squeeze())
        batch = {
            'observations': observations_list,
            'actions': actions_list,
            'next_actions': next_actions_list,
            'rewards': rewards_list,
            'masks': masks_list
        }
        return batch

    def get_traj_start_end_indices(self, buffer_index):
        current_traj_ind = self.dataset_dict['trajectory_id'][buffer_index].squeeze()
        matching_inds = np.where(self.dataset_dict['trajectory_id'] == current_traj_ind)[0].astype(int)
        step = matching_inds[1:] - matching_inds[:-1]
        if np.all(step == 1):
            consecutive = True
        else:
            consecutive = False
        return matching_inds[0], matching_inds[-1], consecutive


    def compute_action_stats(self):
        actions = self.dataset_dict['actions']
        return {'mean': actions.mean(axis=0), 'std': actions.std(axis=0)}

    def normalize_actions(self, action_stats):
        # do not normalize gripper dimension (last dimension)
        copy.deepcopy(action_stats)
        action_stats['mean'][-1] = 0
        action_stats['std'][-1] = 1
        self.dataset_dict['actions'] = (self.dataset_dict['actions'] - action_stats['mean']) / action_stats['std']
        self.dataset_dict['next_actions'] = (self.dataset_dict['next_actions'] - action_stats['mean']) / action_stats['std']

    def get_snap_shot(self):
        return dict(
            dataset_dict=_slice_recursively(self.dataset_dict, self._insert_index),
            _insert_index=self._insert_index,
            _traj_counter=self._traj_counter,
            _size=self._size,
            # for mem-efficient replay:
            _first=self._first,
            _is_correct_index=self._is_correct_index,
            _all_indices=self._all_indices
        )

    def insert_snap_shot(self, data):
        _insert_snapshot_recursively(data['dataset_dict'], self.dataset_dict)
        self._insert_index = data['_insert_index']
        self._traj_counter = data['_traj_counter']
        self._size = data['_size']

        # for mem-effic:
        self._first = data['_first']
        self._is_correct_index = data['_is_correct_index']
        self._all_indices = data['_all_indices']

    def save(self, filename):
        np.save(filename, np.array([self.get_snap_shot()]))

    def restore(self, filename):
        bufferdata = np.load(filename, allow_pickle=True)[0]
        self.insert_snap_shot(bufferdata)
