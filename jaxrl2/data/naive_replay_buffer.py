from os import terminal_size
from typing import Union
from typing import Iterable, Optional
import jax 

import gym
import gym.spaces
import numpy as np

import copy

from jaxrl2.data.dataset import Dataset, DatasetDict
import collections
from flax.core import frozen_dict

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



class NaiveReplayBuffer(Dataset):
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, capacity: int):
        self.observation_space = observation_space
        self.action_space = action_space
        self.capacity = capacity

        observations = _init_replay_dict(self.observation_space, self.capacity)
        next_observations = _init_replay_dict(self.observation_space, self.capacity)
        actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
        next_actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
        rewards = np.empty((self.capacity, ), dtype=np.float32)
        mc_return = np.empty((self.capacity, ), dtype=np.float32)
        masks = np.empty((self.capacity, ), dtype=np.float32)
        trajectory_id = np.empty((self.capacity,), dtype=np.float32)
        dones = np.empty((self.capacity,), dtype=np.float32)

        self.data = {
            'observations': observations,
            'next_observations': next_observations,
            'actions': actions,
            'next_actions': next_actions,
            'rewards': rewards,
            'masks': masks,
            'trajectory_id': trajectory_id,
            'dones': dones,
        }

        self.size = 0
        self._traj_counter = 0
        self._start = 0
        self.traj_bounds = dict()

    def increment_traj_counter(self):
        self.traj_bounds[self._traj_counter] = (self._start, self.size) # [start, end)
        self._start = self.size
        self._traj_counter += 1

    def get_random_trajs(self, num_trajs: int):
        self.which_trajs = np.random.randint(0, self._traj_counter-1, num_trajs)
        observations_list = []
        next_observations_list = []
        mc_rewards_list = []
        actions_list = []
        rewards_list = []
        terminals_list = []
        masks_list = []

        for i in self.which_trajs:
            start, end = self.traj_bounds[i]
            
            # handle this as a dictionary
            obs_dict_curr_traj = dict()
            for k in self.data['observations']:
                obs_dict_curr_traj[k] = self.data['observations'][k][start:end]
            observations_list.append(obs_dict_curr_traj)
            
            next_obs_dict_curr_traj = dict()
            for k in self.data['next_observations']:
                next_obs_dict_curr_traj[k] = self.data['next_observations'][k][start:end]    
            next_observations_list.append(next_obs_dict_curr_traj)
            
            actions_list.append(self.data['actions'][start:end])
            rewards_list.append(self.data['rewards'][start:end])
            terminals_list.append(1-self.data['masks'][start:end])
            masks_list.append(self.data['masks'][start:end])
        
        batch = {
            'observations': observations_list,
            'next_observations': next_observations_list,
            'actions': actions_list,
            'rewards': rewards_list,
            'terminals': terminals_list,
            'masks': masks_list
        }
        return batch
        
    def insert(self, data_dict: DatasetDict):
        if self.size == self.capacity:
            # Double the capacity
            observations = _init_replay_dict(self.observation_space, self.capacity)
            next_observations = _init_replay_dict(self.observation_space, self.capacity)
            actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
            next_actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
            rewards = np.empty((self.capacity, ), dtype=np.float32)
            masks = np.empty((self.capacity, ), dtype=np.float32)

            data_new = {
                'observations': observations,
                'next_observations': next_observations,
                'actions': actions,
                'next_actions': next_actions,
                'rewards': rewards,
                'masks': masks,
            }

            for x in self.data:
                if isinstance(self.data[x], np.ndarray):
                    self.data[x] = np.concatenate((self.data[x], data_new[x]), axis=0)
                elif isinstance(self.data[x], dict):
                    for y in self.data[x]:
                        self.data[x][y] = np.concatenate((self.data[x][y], data_new[x][y]), axis=0)
                else:
                    raise TypeError()
            self.capacity *= 2


        for x in data_dict:
            if x in self.data:
                if isinstance(data_dict[x], dict):
                    for y in data_dict[x]:
                        self.data[x][y][self.size] = data_dict[x][y]
                else:                        
                    self.data[x][self.size] = data_dict[x]
        self.size += 1
    
    def compute_action_stats(self):
        actions = self.data['actions']
        return {'mean': actions.mean(axis=0), 'std': actions.std(axis=0)}

    def normalize_actions(self, action_stats):
        # do not normalize gripper dimension (last dimension)
        copy.deepcopy(action_stats)
        action_stats['mean'][-1] = 0
        action_stats['std'][-1] = 1
        self.data['actions'] = (self.data['actions'] - action_stats['mean']) / action_stats['std']
        self.data['next_actions'] = (self.data['next_actions'] - action_stats['mean']) / action_stats['std']

    def sample(self, batch_size: int, keys: Optional[Iterable[str]] = None, indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:
        indices = np.random.randint(0, self.size, batch_size)
        data_dict = {}
        for x in self.data:
            if isinstance(self.data[x], np.ndarray):
                data_dict[x] = self.data[x][indices]
            elif isinstance(self.data[x], dict):
                data_dict[x] = {}
                for y in self.data[x]:
                    data_dict[x][y] = self.data[x][y][indices]
            else:
                raise TypeError()
        
        return frozen_dict.freeze(data_dict)

    def get_iterator(self, batch_size: int, keys: Optional[Iterable[str]] = None, indx: Optional[np.ndarray] = None, queue_size: int = 2):
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


            
class NaiveReplayBufferParallel(NaiveReplayBuffer):
    """
    Implements naive buffer with parallelism
    """
    def __init__(self, observation_space: gym.Space, action_space: gym.Space,
                 capacity: int, num_devices=len(jax.devices())):
        self.num_devices = num_devices
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         capacity=capacity)
        
    def get_iterator(self,
                     batch_size: int,
                     keys: Optional[Iterable[str]] = None,
                     indx: Optional[np.ndarray] = None,
                     queue_size: int = 2):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            assert batch_size % self.num_devices == 0
            effective_batch_size = batch_size // self.num_devices
            for _ in range(n):
                data = [self.sample(effective_batch_size, keys, indx) for _ in range(self.num_devices)]   
                queue.append(jax.device_put_sharded(data, jax.devices()))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)
