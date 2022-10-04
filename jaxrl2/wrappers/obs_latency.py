import collections

import gym
import numpy as np
from gym.spaces import Box
from gym.spaces import Dict



class ObsLatency(gym.Wrapper):
    def __init__(self, env, latency: int):
        super().__init__(env)
        self._latency = latency

        self._pixels = collections.deque(maxlen=latency + 1)

        self._reward_frames = collections.deque(maxlen=latency + 1)
        self._terminal_frames = collections.deque(maxlen=latency + 1)

    def reset(self):
        obs = self.env.reset()
        for i in range(self._latency):
            self._pixels.append(obs['pixels'])
            self._reward_frames.append(0.0)
            self._terminal_frames.append(False)
        obs['pixels'] = self._pixels[0]
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._pixels.append(obs['pixels'])
        self._reward_frames.append(reward)
        self._terminal_frames.append(done)

        # For debugging purposes only, one can actually print or log these
        # this info is returned to the final algorithm main runner loop
        info['orig_obs'] = obs
        info['orig_reward'] = reward
        info['orig_terminal'] = done

        obs['pixels'] = self._pixels[0]
        return obs, self._reward_frames[0], self._terminal_frames[0], info