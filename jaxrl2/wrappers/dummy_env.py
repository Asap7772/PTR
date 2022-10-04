from gym import Env
import gym.spaces
import numpy as np
from gym.spaces import Box, Dict

# class DummyEnv(Env):
#     def __init__(self, image_shape = (128, 128, 3), state_shape=(3,), action_shape=(4,)) -> None:
#         super().__init__()

#         self.image_shape = image_shape
#         self.state_shape = state_shape
#         self.action_shape = action_shape

#         self.observation_space = gym.spaces.Dict({
#                 'image': gym.spaces.Box(0, 255, self.image_shape),
#                 'state': gym.spaces.Box(-1, 1, self.state_shape)
#             })

#         self.action_space = gym.spaces.Box(-1, 1, self.action_shape)

#     def step(self, action):
#         return {
#             'image': np.zeros(self.image_shape).flatten(),
#             'state': np.zeros(self.state_shape)
#         }, 0, 0, {}

#     def reset(self):
#         return {
#             'image': np.zeros(self.image_shape).flatten(),
#             'state': np.zeros(self.state_shape)
#         }

# class DummyKitchenEnv(DummyEnv):
#     def __init__(self, image_shape=(128,128,3), state_shape=(3,), action_shape=(4,), normalized_actions=True) -> None:
#         super().__init__(image_shape, state_shape, action_shape)

#         self.observation_space = Box(low=0,
#                                      high=255,
#                                      shape=image_shape,
#                                      dtype=np.uint8)

#         if normalized_actions:
#             self.action_space = gym.spaces.Box(low=np.array([-10, -10, -10, -10, -10, -10, 0]), 
#                                            high=np.array([10, 10, 10, 10, 10, 10, 1]))
#         else:
#             self.action_space = gym.spaces.Box(low=np.array([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0]), 
#                                            high=np.array([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1]))


class DummyEnv():
    def __init__(self, add_states=False, num_tasks=1):
        super().__init__()
        obs_dict = dict(pixels=Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8))
        if add_states:
            obs_dict['state'] = Box(low=-100000, high=100000, shape=(7,), dtype=np.float32)
        if num_tasks > 1:
            obs_dict['task_id'] = Box(low=0, high=1, shape=(num_tasks,), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        self.spec = None
        self.action_space = Box(
            np.asarray([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.]),
            np.asarray([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0]),
            dtype=np.float32)
        
    def set_action_stats(*args, **kwargs):
        pass

    def seed(self, seed):
        pass

