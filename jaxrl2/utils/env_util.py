
from jaxrl2.wrappers import FrameStack
from jaxrl2.wrappers.reaching_reward_wrapper import ReachingReward
from jaxrl2.wrappers.prev_action_wrapper import PrevActionStack
from jaxrl2.wrappers.state_wrapper import StateStack
from jaxrl2.wrappers.rescale_actions_wrapper import RescaleActions
from jaxrl2.wrappers.normalize_actions_wrapper import NormalizeActions

TARGET_POINT = np.array([0.28425417, 0.04540814, 0.07545623])  # mean
# TARGET_POINT = np.array([0.23, 0., 0.1])

def wrap(env, variant):
    assert not (variant.normalize_actions and variant.rescale_actions)
    if variant.reward_type == 'dense':
        env = ReachingReward(env, TARGET_POINT, variant.reward_type)
    if variant.add_prev_actions:
        if variant.frame_stack == 1:
            num_action_stack = 1
        else:
            num_action_stack = variant.frame_stack - 1
        env = PrevActionStack(env, num_action_stack)
    if variant.rescale_actions:
        print ('Rescaling actions in environment..............')
        env = RescaleActions(env)
    elif variant.normalize_actions:
        print ('Normalizing actions in environment.............')
        env = NormalizeActions(env)
    if variant.add_states:
        env = StateStack(env, variant.frame_stack)
    if not variant.from_states:
        env = FrameStack(env, variant.frame_stack)
    env = gym.wrappers.TimeLimit(env, variant.episode_timelimit)
    return env