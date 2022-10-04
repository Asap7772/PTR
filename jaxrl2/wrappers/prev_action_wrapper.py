import collections

import gym
import numpy as np
from gym.spaces import Box


class PrevActionStack(gym.Wrapper):

    def __init__(self, env, num_stack: int):
        super().__init__(env)
        self._num_stack = num_stack

        prev_action_stack_spaces = Box(
            low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32)

        self._env_dim = prev_action_stack_spaces.shape[0]

        low = np.repeat(prev_action_stack_spaces.low[..., np.newaxis],
                        num_stack,
                        axis=-1)
        high = np.repeat(prev_action_stack_spaces.high[..., np.newaxis],
                         num_stack,
                         axis=-1)
        new_action_stack_spaces = Box(low=low,
                                   high=high,
                                   dtype=prev_action_stack_spaces.dtype)
        self.observation_space.spaces['prev_action'] = new_action_stack_spaces

        self._action_frames = collections.deque(maxlen=num_stack)

    def reset(self):
        next_obs = self.env.reset()
        # At reset pass in all zeros previous actions
        for i in range(self._num_stack):
            self._action_frames.append(np.zeros(self._env_dim))
        next_obs['prev_action'] = self.action_frames[None]
        return next_obs

    @property
    def action_frames(self):
        return np.stack(self._action_frames, axis=-1)

    def step(self, action):
        next_obs, reward, done, info = super().step(action)
        if isinstance(action, dict):
            action = action['action'].squeeze()
        self._action_frames.append(action)
        next_obs['prev_action'] = self.action_frames[None]
        return next_obs, reward, done, info

    def observation(self, observation):
        print ('Going through action stacking')
        return {
            'pixels': observation['pixels'],
            'state': observation['state'],
            'prev_action': observation['prev_action']
        } 