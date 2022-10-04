import collections
import copy
import glob
from typing import Iterable, Optional

import gym
import jax
import numpy as np
from flax.core import frozen_dict
from gym.spaces import Box

from jaxrl2.data.dataset import DatasetDict, _sample
from jaxrl2.data.replay_buffer import ReplayBuffer


class OfflineMemoryEfficientReplayBuffer(ReplayBuffer):

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 capacity: int,
                 data_url: str = '/home/rafael/IRIS/vd5rl/EPISODESSHORT/',
                 replay: int = 10):

        self._data_url = data_url
        self._replay = replay
        pixel_obs_space = observation_space.spaces['pixels']
        self._num_stack = pixel_obs_space.shape[-1]
        self._unstacked_dim_size = pixel_obs_space.shape[-2]
        low = pixel_obs_space.low[..., 0]
        high = pixel_obs_space.high[..., 0]
        unstacked_pixel_obs_space = Box(low=low,
                                        high=high,
                                        dtype=pixel_obs_space.dtype)
        observation_space = copy.deepcopy(observation_space)
        observation_space.spaces['pixels'] = unstacked_pixel_obs_space

        self._first = True
        self._correct_index = np.full(capacity, False, dtype=bool)

        next_observation_space_dict = copy.deepcopy(observation_space.spaces)
        next_observation_space_dict.pop('pixels')
        next_observation_space = gym.spaces.Dict(next_observation_space_dict)

        super().__init__(observation_space,
                         action_space,
                         capacity,
                         next_observation_space=next_observation_space)

    def load(self):
        while self._size < self._capacity:
            self.load_episode()

    def load_episode(self):
        file = self.np_random.choice(glob.glob(self._data_url + '*.pkl'))
        episode = np.load(file, allow_pickle=True)
        frames = collections.deque(maxlen=self._num_stack)
        for _ in range(self._num_stack):
            frames.append(
                np.concatenate([
                    episode['camera_0_rgb'][0], episode['camera_1_rgb'][0],
                    episode['camera_gripper_rgb'][0]
                ],
                               axis=-1))
        for t in range(episode['reward'].shape[0] - 1):
            transition = dict()
            transition['observations'] = dict()
            transition['observations']['pixels'] = np.stack(frames, axis=-1)
            transition['observations']['ee_forces'] = episode['ee_forces'][t]
            transition['observations']['ee_qp'] = episode['ee_qp'][t]
            transition['observations']['robot_qp'] = episode['robot_qp'][t]

            transition['actions'] = episode['action'][t + 1]
            transition['rewards'] = episode['reward'][t + 1]

            frames.append(
                np.concatenate([
                    episode['camera_0_rgb'][t + 1],
                    episode['camera_1_rgb'][t + 1],
                    episode['camera_gripper_rgb'][t + 1]
                ],
                               axis=-1))

            transition['next_observations'] = dict()
            transition['next_observations']['pixels'] = np.stack(frames,
                                                                 axis=-1)
            transition['next_observations']['ee_forces'] = episode[
                'ee_forces'][t + 1]
            transition['next_observations']['ee_qp'] = episode['ee_qp'][t + 1]
            transition['next_observations']['robot_qp'] = episode['robot_qp'][
                t + 1]

            transition['masks'] = 0.0
            transition['dones'] = 0.0

            self.insert(transition)

    def insert(self, data_dict: DatasetDict):
        if self._insert_index == 0 and self._capacity == len(
                self) and not self._first:
            indxs = np.arange(len(self) - self._num_stack, len(self))
            for indx in indxs:
                element = super().sample(1, indx=indx)
                self._correct_index[self._insert_index] = False
                super().insert(element)

        data_dict = data_dict.copy()
        data_dict['observations'] = data_dict['observations'].copy()
        data_dict['next_observations'] = data_dict['next_observations'].copy()

        obs_pixels = data_dict['observations'].pop('pixels')
        next_obs_pixels = data_dict['next_observations'].pop('pixels')

        if self._first:
            for i in range(self._num_stack):
                data_dict['observations']['pixels'] = obs_pixels[..., i]
                self._correct_index[self._insert_index] = False
                super().insert(data_dict)

        data_dict['observations']['pixels'] = next_obs_pixels[..., -1]

        self._first = data_dict['dones']

        self._correct_index[self._insert_index] = True
        super().insert(data_dict)

        for i in range(self._num_stack):
            indx = (self._insert_index + i) % len(self)
            self._correct_index[indx] = False

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:

        if indx is None:
            indx = np.empty(batch_size, dtype=int)
            for i in range(batch_size):
                while True:
                    if hasattr(self.np_random, 'integers'):
                        indx[i] = self.np_random.integers(
                            self._num_stack, len(self))
                    else:
                        indx[i] = self.np_random.randint(
                            self._num_stack, len(self))
                    if self._correct_index[indx[i]]:
                        break
        else:
            raise ValueError()

        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()
        else:
            raise ValueError()

        keys = list(keys)
        keys.remove('observations')

        batch = super().sample(batch_size, keys, indx)
        batch = batch.unfreeze()

        obs_keys = self.dataset_dict['observations'].keys()
        obs_keys = list(obs_keys)
        obs_keys.remove('pixels')

        batch['observations'] = {}
        for k in obs_keys:
            batch['observations'][k] = _sample(
                self.dataset_dict['observations'][k], indx)

        obs_pixels = self.dataset_dict['observations']['pixels']
        obs_pixels = np.lib.stride_tricks.sliding_window_view(obs_pixels,
                                                              self._num_stack +
                                                              1,
                                                              axis=0)
        obs_pixels = obs_pixels[indx - self._num_stack]
        batch['observations']['pixels'] = obs_pixels

        return frozen_dict.freeze(batch)

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
            for _ in range(self._replay):
                yield queue.popleft()
                enqueue(1)
            self.load_episode()
