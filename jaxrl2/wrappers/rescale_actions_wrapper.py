import collections

import gym
import numpy as np
from gym.spaces import Box


class RescaleActions(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        act_high = np.ones(7)
        self.orig_low = env.action_space.low
        self.orig_high = env.action_space.high
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def step(self, action):
        if isinstance(action, dict):
            action['action'] = self.unnormalize_actions(action['action'])
        else:
            action = self.unnormalize_actions(action)
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def rescale_actions(self, actions, safety_margin=0.01):
        """
        rescale xyz, and rotation actions to be within -1 and 1, then clip actions to stay within safety margin
        used when loading unnormalized actions into the replay buffer (the np files store unnormalized actions)
        """
        actions = actions.squeeze()
        assert actions.shape == (7,)

        resc_actions = (actions - self.orig_low) / (self.orig_high - self.orig_low) * 2 - 1
        return np.clip(resc_actions, -1 + safety_margin, 1 - safety_margin)

    def unnormalize_actions(self, actions):
        """
        rescale xyz, and rotation actions to be within the original environments bounds e.g. +-0.05 for xyz, +-0.25 for rotations
        """
        assert actions.shape == (1,7)
        actions = actions.squeeze()
        actions_rescaled = (actions + 1) / 2 * (self.orig_high - self.orig_low) + self.orig_low

        if np.any(actions_rescaled > self.orig_high):
            print('action bounds violated: ', actions)
        if np.any(actions_rescaled < self.orig_low):
            print('action bounds violated: ', actions)
        return actions_rescaled[None]
