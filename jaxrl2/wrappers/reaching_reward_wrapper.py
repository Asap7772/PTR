import collections

import gym
import numpy as np
from gym.spaces import Box


def compute_distance_reward(state, target_point, reward_type):
    goal_dist = np.linalg.norm(state - target_point)
    if reward_type == 'dense':
        reward = -goal_dist * 100
    elif reward_type == 'sparse':
        if goal_dist < 0.04:
            reward = 100
        else:
            reward = -1
    elif reward_type == 'dense_sparse':
        reward = -goal_dist * 100
        if goal_dist < 0.04:
            reward = 100
    else:
        raise ValueError('reward type not found')
    return reward

class ReachingReward(gym.Wrapper):

    def __init__(self, env, target_point, reward_type):
        self.target = target_point
        self.reward_type = reward_type
        super().__init__(env)

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        #  for real robot keep gripper open for reaching tasks
        if isinstance(action, dict):
            action_vec = action['action']
            action_vec[..., -1] = 1
            action['action'] = action_vec

        obs, _, done, info = self.env.step(action)
        xyz = obs['state'].squeeze()[:3]
        reward = compute_distance_reward(xyz, self.target, self.reward_type)
        # print('_____________________________________________')
        # print('distance reward ', reward)
        # print('target ', self.target)
        # print('current  ', xyz)
        return obs, reward, done, info