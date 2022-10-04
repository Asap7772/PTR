import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper

from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.repeat_action import RepeatAction
from jaxrl2.wrappers.universal_seed import UniversalSeed


def wrap_pixels(env: gym.Env,
                action_repeat: int,
                image_size: int = 84,
                num_stack: int = 3,
                camera_id: int = 0) -> gym.Env:
    if action_repeat > 1:
        env = RepeatAction(env, action_repeat)

    env = UniversalSeed(env)
    env = gym.wrappers.RescaleAction(env, -1, 1)

    env = PixelObservationWrapper(env,
                                  pixels_only=True,
                                  render_kwargs={
                                      'pixels': {
                                          'height': image_size,
                                          'width': image_size,
                                          'camera_id': camera_id
                                      }
                                  })

    if num_stack > 1:
        env = FrameStack(env, num_stack=num_stack)

    return env
