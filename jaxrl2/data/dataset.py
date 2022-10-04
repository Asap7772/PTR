from typing import Dict, Iterable, Optional, Tuple, Union
import collections
import jax
import numpy as np
from gym.utils import seeding

from jaxrl2.types import DataType

DatasetDict = Dict[str, DataType]
from flax.core import frozen_dict

def concat_recursive(batches):
    new_batch = {}
    for k, v in batches[0].items():
        if isinstance(v, frozen_dict.FrozenDict):
            new_batch[k] = concat_recursive([batches[0][k], batches[1][k]])
        else:
            new_batch[k] = np.concatenate([b[k] for b in batches], 0)
    return new_batch

def _check_lengths(dataset_dict: DatasetDict,
                   dataset_len: Optional[int] = None) -> int:
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, np.ndarray):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, 'Inconsistent item lengths in the dataset.'
        else:
            raise TypeError('Unsupported type.')
    return dataset_len


def _split(dataset_dict: DatasetDict,
           index: int) -> Tuple[DatasetDict, DatasetDict]:
    train_dataset_dict, test_dataset_dict = {}, {}
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            train_v, test_v = _split(v, index)
        elif isinstance(v, np.ndarray):
            train_v, test_v = v[:index], v[index:]
        else:
            raise TypeError('Unsupported type.')
        train_dataset_dict[k] = train_v
        test_dataset_dict[k] = test_v
    return train_dataset_dict, test_dataset_dict


def _sample(dataset_dict: Union[np.ndarray, DatasetDict],
            indx: np.ndarray) -> DatasetDict:
    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[indx]
    elif isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, indx)
    else:
        raise TypeError("Unsupported type.")
    return batch


class Dataset(object):

    def __init__(self, dataset_dict: DatasetDict, seed: Optional[int] = None):
        self.dataset_dict = dataset_dict
        self.dataset_len = _check_lengths(dataset_dict)

        # Seeding similar to OpenAI Gym:
        # https://github.com/openai/gym/blob/master/gym/spaces/space.py#L46
        self._np_random = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self) -> np.random.RandomState:
        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: Optional[int] = None) -> list:
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def __len__(self) -> int:
        return self.dataset_len

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:
        if indx is None:
            if hasattr(self.np_random, 'integers'):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], indx)
            else:
                batch[k] = self.dataset_dict[k][indx]

        return frozen_dict.freeze(batch)

    def split(self, ratio: float) -> Tuple['Dataset', 'Dataset']:
        assert 0 < ratio and ratio < 1
        index = int(self.dataset_len * ratio)
        train_dataset_dict, test_dataset_dict = _split(self.dataset_dict,
                                                       index)
        return Dataset(train_dataset_dict), Dataset(test_dataset_dict)


class MixingReplayBuffer():

    def __init__(
            self,
            replay_buffers,
            mixing_ratio
    ):

        """
        :param replay_buffers: sample from given replay buffer with specified probability
        """

        self.replay_buffers = replay_buffers
        self.mixing_ratio = mixing_ratio
        assert len(replay_buffers) == 2

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:

        batches = []
        size_first = int(np.floor(batch_size*self.mixing_ratio))
        sub_batch_sizes = [size_first, batch_size - size_first]
        for buf, sb in zip(self.replay_buffers, sub_batch_sizes):
            batches.append(buf.sample(sb))


        return frozen_dict.freeze(concat_recursive(batches))

    def set_mixing_ratio(self, mixing_ratio):
        self.mixing_ratio = mixing_ratio

    def seed(self, seed):
        [b.seed(seed) for b in self.replay_buffers]

    def length(self):
        return [b.length() for b in self.replay_buffers]

    def get_random_trajs(self, batch_size):
        return [b.get_random_trajs(batch_size) for b in self.replay_buffers]


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
    
    def increment_traj_counter(self):
        [b.increment_traj_counter() for b in self.replay_buffers]

    def compute_action_stats(self):

        action_stats_0 = self.replay_buffers[0].compute_action_stats()
        action_stats_1 = self.replay_buffers[1].compute_action_stats()

        ratio = self.mixing_ratio
        actions_mean = ratio * action_stats_0['mean'] + (1 - ratio) * action_stats_1['mean']
        actions_std = np.sqrt(ratio * action_stats_0['std'] ** 2 + (1 - ratio) * action_stats_1['std']** 2 + ratio * (1 - ratio) * (action_stats_0['mean'] - action_stats_1['mean']) ** 2)

        return {'mean': actions_mean, 'std': actions_std}

    def normalize_actions(self, action_stats):
        # do not normalize gripper dimension (last dimension)
        [b.normalize_actions(action_stats) for b in self.replay_buffers]
        
class MixingReplayBufferParallel():
    
    def __init__(
            self,
            replay_buffers,
            mixing_ratio,
            num_devices=len(jax.devices())
    ):

        """
        :param replay_buffers: sample from given replay buffer with specified probability
        """

        self.replay_buffers = replay_buffers
        self.mixing_ratio = mixing_ratio
        assert len(replay_buffers) == 2
        self.num_devices=num_devices

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:

        batches = []
        size_first = int(np.floor(batch_size*self.mixing_ratio))
        sub_batch_sizes = [size_first, batch_size - size_first]
        for buf, sb in zip(self.replay_buffers, sub_batch_sizes):
            batches.append(buf.sample(sb))


        return frozen_dict.freeze(concat_recursive(batches))

    def set_mixing_ratio(self, mixing_ratio):
        self.mixing_ratio = mixing_ratio

    def seed(self, seed):
        [b.seed(seed) for b in self.replay_buffers]

    def length(self):
        return [b.length() for b in self.replay_buffers]

    def get_random_trajs(self, batch_size):
        return [b.get_random_trajs(batch_size) for b in self.replay_buffers]


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
    
    def increment_traj_counter(self):
        [b.increment_traj_counter() for b in self.replay_buffers]

    def compute_action_stats(self):

        action_stats_0 = self.replay_buffers[0].compute_action_stats()
        action_stats_1 = self.replay_buffers[1].compute_action_stats()

        ratio = self.mixing_ratio
        actions_mean = ratio * action_stats_0['mean'] + (1 - ratio) * action_stats_1['mean']
        actions_std = np.sqrt(ratio * action_stats_0['std'] ** 2 + (1 - ratio) * action_stats_1['std']** 2 + ratio * (1 - ratio) * (action_stats_0['mean'] - action_stats_1['mean']) ** 2)

        return {'mean': actions_mean, 'std': actions_std}

    def normalize_actions(self, action_stats):
        # do not normalize gripper dimension (last dimension)
        [b.normalize_actions(action_stats) for b in self.replay_buffers]
