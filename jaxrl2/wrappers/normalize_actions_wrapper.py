import collections

import gym
import numpy as np
from gym.spaces import Box


class NormalizeActions(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.mean = None
        self.std = None

    def set_action_stats(self, action_stats):
        self.mean = action_stats['mean']
        self.std = action_stats['std']

    def step(self, action):
        if isinstance(action, dict):
            action['action'] = self.unnormalize_actions(action['action'])
        else:
            action = self.unnormalize_actions(action)
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def normalize_actions(self, actions):
        actions = actions.squeeze()
        assert actions.shape == (7,)
        actions = (actions - self.mean) / self.std
        return actions

    def unnormalize_actions(self, actions):
        """
        rescale xyz, and rotation actions to be within the original environments bounds e.g. +-0.05 for xyz, +-0.25 for rotations
        """
        actions = actions.squeeze()
        assert actions.shape == (7,)
        actions = actions * self.std + self.mean
        return actions[None]
