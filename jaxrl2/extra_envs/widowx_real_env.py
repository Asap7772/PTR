import pdb

import numpy as np
import os
from widowx_envs.widowx.widowx_env import WidowXEnv
from gym.spaces import Dict
from gym.spaces import Box
import time
from jaxrl2.utils.visualization_utils import sigmoid
import rospy
import random
import pickle as pkl

from widowx_envs.utils.exceptions import Environment_Exception


from examples.train_pixels_real import TARGET_POINT

import glob

traj_group_open = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/open_microwave/*/raw/traj_group*/traj*')
traj_group_close = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/close_microwave/*/raw/traj_group*/traj*')
traj_group_sushi = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2/put_sushi_in_pot_cardboard_fence/*/raw/traj_group*/traj*')
traj_group_sushi_out = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2/take_sushi_out_of_pot_cardboard_fence/*/raw/traj_group*/traj*')
traj_group_bowlplate = glob.glob(os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2/put_bowl_on_plate_cardboard_fence/*/raw/traj_group*/traj*')

start_transforms = dict(
    right_front = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_11-06-56/raw/traj_group0/traj0', 150],
    middle = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_11-06-56/raw/traj_group0/traj1', 290],
    left_front = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_11-06-56/raw/traj_group0/traj2', 200],
    left_rear = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_17-54-21/raw/traj_group0/traj0', 290],
    right_rear = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_17-54-21/raw/traj_group0/traj1', 290],

    openmicrowave = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/open_microwave/2021-12-01_16-14-39/raw/traj_group0/traj0', 7],
    closemicrowave = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/close_microwave/2021-12-02_12-14-59/raw/traj_group0/traj1', 0],
    openmicrowave_sampled = [traj_group_open, 10],
    # openclosemicrowave_sampled = {3: [traj_group_open, 10], 1:[traj_group_close, (10, 16)]},  # use without aliasing
    openclosemicrowave_sampled = {1: [traj_group_open, 10], 0:[traj_group_close, (10, 16)]},  # with aliasing
    reaching = [os.environ['DATA'] + '/robonetv2/online_datacollection/online_reaching/berkeley/toykitchen1/pix_policy_std0.3_initscale0.0_collectdata/2022-04-25_18-10-03/raw/traj_group0/traj0', 0],
    # toykitchen2_sushi_cardboard_sampled = [traj_group_sushi, (5, 12)]
    toykitchen2_sushi_cardboard_sampled = [traj_group_sushi, (0, 5)],
    toykitchen2_sushi_out_cardboard_sampled = [traj_group_sushi_out, (0, 5)],
    toykitchen2_bowlplate_cardboard_sampled= [traj_group_bowlplate, (0, 5)],
)

def get_env_params(variant):
    env_params = {
        'fix_zangle': True,  # do not apply random rotations to start state
        'move_duration': 0.2,
        'adaptive_wait': True,
        'move_to_rand_start_freq': variant.move_to_rand_start_freq if 'move_to_rand_start_freq' in variant else 1,
        # 'override_workspace_boundaries': [[0.17, -0.08, 0.06, -1.57, 0], [0.35, 0.08, 0.1, 1.57, 0]],
        # broad action boundaries for reaching
        # 'override_workspace_boundaries': [[0.100, - 0.1820, 0.0, -1.57, 0], [0.40, 0.143, 0.24, 1.57, 0]],
        # broad action boundaries for door opening
        'override_workspace_boundaries': [[0.100, -0.25, 0.0, -1.57, 0], [0.41, 0.143, 0.33, 1.57, 0]],

        'action_clipping': 'xyz',
        'catch_environment_except': True,
        'target_point': TARGET_POINT,
        'add_states': variant.add_states,
        'from_states': variant.from_states,
        'reward_type': variant.reward_type,
        'start_transform': None if variant.start_transform == '' else start_transforms[variant.start_transform],
        'randomize_initpos': 'full_area'
    }
    return env_params


class JaxRLWidowXEnv(WidowXEnv):
    def __init__(self, env_params=None, task_id=None, num_tasks=None, fixed_image_size=128,
                 control_viewpoint=0 # used for reward function
                 ):

        super().__init__(env_params)
        self.image_size = fixed_image_size
        self.task_id = task_id
        self.num_tasks = num_tasks

        obs_dict = {}
        if not self._hp.from_states:
            obs_dict['pixels'] = Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        if self._hp.add_states:
            obs_dict['state'] = Box(low=-100000, high=100000, shape=(7,), dtype=np.float32)
        if self._hp.add_task_id:
            obs_dict['task_id'] = Box(low=0, high=1, shape=(num_tasks,), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        self.move_except = False
        self.control_viewpoint = control_viewpoint
        self.spec = None
        self.requires_timed = True
        self.do_render = True
        self.traj_counter = 0

    def _default_hparams(self):
        from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
        default_dict = {
            'gripper_attached': 'custom',
            'skip_move_to_neutral': True,
            'camera_topics': [IMTopic('/cam0/image_raw')],
            'image_crop_xywh': None,  # can be a tuple like (0, 0, 100, 100)
            'add_states': False,
            'add_task_id':False,
            'from_states': False,
            'reward_type': None
        }
        parent_params = super()._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def reset(self, itraj=None, reset_state=None):
        if itraj is None:
            itraj = self.traj_counter
        self.traj_counter += 1
        return super().reset(itraj, reset_state)


    def _get_processed_image(self, image=None):
        from skimage.transform import resize
        downsampled_trimmed_image = resize(image, (self.image_size, self.image_size), anti_aliasing=True, preserve_range=True).astype(np.uint8)
        return downsampled_trimmed_image

    def step(self, action):
        obs = super().step(action['action'].squeeze(), action['tstamp_return_obs'], blocking=False)
        reward = 0
        done = obs['full_obs']['env_done']  # done can come from VR buttons
        info = {}
        if self.move_except:
            done = True
            info['Error.truncated'] = True
            # self.move_to_startstate()
        return obs, reward, done, info

    def disable_render(self):
        self.do_render = False

    def enable_render(self):
        self.do_render = True

    def _get_obs(self):
        full_obs = super()._get_obs()
        obs = {}
        if self.do_render:
            processed_images = np.stack([self._get_processed_image(im) for im in full_obs['images']], axis=0)
            obs['pixels'] = processed_images
        obs['full_obs'] = full_obs
        if self._hp.add_states:
            obs['state'] = self.get_full_state()[None]
        return obs

    def set_task_id(self, task_id):
        self.task_id = task_id

    def get_task_id_vec(self, task_id):
        task_id_vec = None
        if (task_id is not None) and self.num_tasks:
            task_id_vec = np.zeros(self.num_tasks, dtype=np.float32)[None]
            task_id_vec[:, task_id] = 1.0
        return task_id_vec

    def move_to_startstate(self, start_state=None):
        # sel_task = self.select_task_from_reward_function()
        paths, tstep = self._hp.start_transform

        successful = False
        itrial = 0
        print('entering move to startstate loop.')
        while not successful:
            print('move to startstate trial ', itrial)
            itrial += 1
            if itrial > 10:
                import pdb; pdb.set_trace()
            try:
                if not isinstance(paths, str):
                    print(f'sampling random start state from {len(paths)} paths ...')
                    sel_path = random.choice(paths)
                    if isinstance(tstep, int):
                        sel_tstep = np.random.randint(0, tstep)
                    elif isinstance(tstep, tuple) and len(tstep) == 2:
                        sel_tstep = np.random.randint(tstep[0], tstep[1])
                    else:
                        raise ValueError('Incorrect tstep index')
                    print('loading starttransform from {} at step {}'.format(sel_path, sel_tstep))
                else:
                    sel_path = paths
                    sel_tstep = tstep
                transform = pkl.load(open(sel_path + '/obs_dict.pkl', 'rb'))['eef_transform'][sel_tstep]
                self.controller.move_to_starteep(transform, duration=0.8)
                successful = True
            except Environment_Exception:
                self.move_to_neutral()
        
        import pdb; pdb.set_trace()

    def select_task_from_reward_function(self):
        obs = self._get_obs()
        obs_reward = obs.copy()
        obs_reward.pop('full_obs')
        obs_reward['pixels'] = np.expand_dims(obs_reward['pixels'], len(obs_reward['pixels'].shape))

        rewards = []
        for task_id in self.all_tasks:
            obs_reward['task_id'] = self.get_task_id_vec(task_id)
            if 'state' in obs_reward:  # reward classifiers never use proprioceptive states
                obs_reward.pop('state')
            reward = sigmoid(self.reward_function.eval_actions(obs_reward))
            rewards.append(reward)
            print('reward for task {} {}'.format(self.taskid2string[task_id], reward))

        sel_task = self.all_tasks[np.argmin(rewards)]
        print('selected task {} {}'.format(sel_task, self.taskid2string[sel_task]))
        return sel_task

class ImageReachingJaxRLWidowXEnv(JaxRLWidowXEnv):
    def _get_obs(self):
        full_obs = WidowXEnv._get_obs(self)
        obs = {}
        processed_images = np.stack([self._get_processed_image(im) for im in full_obs['images']], axis=0)
        obs['pixels'] = processed_images
        obs['full_obs'] = full_obs
        obs['state'] = self.get_full_state()[None]
        return obs

class BridgeDataJaxRLWidowXRewardAdapter(JaxRLWidowXEnv):
    def __init__(self, env_params=None, reward_function=None, task_id=None, num_tasks=None, fixed_image_size=128,
                 control_viewpoint=0, all_tasks=None, task_id_mapping=None # used for reward function
                 ):

        super().__init__(env_params, task_id, num_tasks, fixed_image_size, control_viewpoint)
        self.reward_function = reward_function
        self.all_tasks = all_tasks
        self.task_string2id = task_id_mapping
        self.taskid2string = {v: k for k, v in self.task_string2id.items()}

    def set_reward_function(self, reward_function):
        self.reward_function = reward_function

    def set_task_id(self, task_id):
        self.task_id = task_id

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.reward_function:
            reward = self.infer_reward_for_task(obs, self.task_id)
            # for tsk in self.all_tasks:
            #     self.infer_reward_for_task(obs, tsk)

        return obs, reward, done, info

    def infer_reward_for_task(self, obs, task_id):
        task_id_vec = self.get_task_id_vec(task_id)
        obs_reward = obs.copy()
        obs_reward.pop('full_obs')
        obs_reward['task_id'] = task_id_vec
        obs['task_id'] = task_id_vec
        obs_reward['pixels'] = np.expand_dims(obs_reward['pixels'], len(obs_reward['pixels'].shape))
        if 'state' in obs_reward:  # reward classifiers never use proprioceptive states
            obs_reward.pop('state')
        # print('obs reward shape', obs_reward['pixels'].shape)
        t0 = time.time()
        reward = self.reward_function.eval_actions(obs_reward)
        reward = sigmoid(reward)
        # print('infer rewrard took ', time.time() - t0)
        print("Predicted reward for task {} {}".format(self.taskid2string[task_id], reward))
        reward = np.asarray(reward)[0]
        return reward


class VR_JaxRLWidowXEnv(JaxRLWidowXEnv):
    def __init__(self, env_params=None, task_id=None, num_tasks=None, fixed_image_size=128,
                 control_viewpoint=0 # used for reward function
                 ):

        super().__init__(env_params, task_id, num_tasks, fixed_image_size, control_viewpoint)

        from oculus_reader import OculusReader
        self.oculus_reader = OculusReader()

    def get_vr_buttons(self):
        poses, buttons = self.oculus_reader.get_transformations_and_buttons()
        if 'RG' in buttons:
            buttons['handle'] = buttons['RG']
        else:
            buttons['handle'] = False
        return buttons

    def _default_hparams(self):
        default_dict = {
            'num_task_stages': 1,
            'make_oculus_reader': True

        }
        parent_params = super(VR_JaxRLWidowXEnv, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def step(self, action):
        """
        :param action:  endeffector velocities
        :return:  observations
        """
        obs, reward, done, info = super(VR_JaxRLWidowXEnv, self).step(action)
        if self.get_vr_buttons()['B']:
            done = True
        return obs, reward, done, info

    def reset(self, itraj=None, reset_state=None):
        obs = super(VR_JaxRLWidowXEnv, self).reset(itraj=itraj)
        start_key = 'handle'
        print('waiting for {} button press to start recording. Press B to go to neutral.'.format(start_key))
        buttons = self.get_vr_buttons()
        while not buttons[start_key]:
            buttons = self.get_vr_buttons()
            rospy.sleep(0.01)
            if 'B' in buttons and buttons['B']:
                self.move_to_neutral()
                print("moved to neutral. waiting for {} button press to start recording.".format(start_key))
        return self._get_obs()

    def ask_confirmation(self):
        print('current endeffector pos', self.get_full_state()[:3])
        print('current joint angles pos', self.controller.get_joint_angles())
        print('Was the trajectory okay? Press A to confirm and RJ to discard')
        buttons = self.get_vr_buttons()
        while not buttons['A'] and not buttons['RJ']:
            buttons = self.get_vr_buttons()
            rospy.sleep(0.01)

        if buttons['RJ']:
            print('trajectory discarded!')
            return False
        if buttons['A']:
            print('trajectory accepted!')
            return True

class VR_JaxRLWidowXEnv_DAgger(VR_JaxRLWidowXEnv):
    def ask_confirmation(self):
        print('Was the trajectory okay? Press A to save a successful trajectory, trigger to save an unsuccessful trajectory, and RJ to discard')
        buttons = self.get_vr_buttons()

        while buttons['A'] or buttons['RJ'] or buttons['RTr']:
            buttons = self.get_vr_buttons()
            rospy.sleep(0.01)

        while not buttons['A'] and not buttons['RJ'] and not buttons['RTr']:
            buttons = self.get_vr_buttons()
            rospy.sleep(0.01)

        if buttons['RJ']:
            print('trajectory discarded!')
            return False
        elif buttons['A']:
            print('successful trajectory accepted!')
            return 'Success'
        elif buttons['RTr']:
            print('unsuccessful trajectory accepted!')
            return 'Failure'

class BridgeDataJaxRLVRWidowXReward(BridgeDataJaxRLWidowXRewardAdapter, VR_JaxRLWidowXEnv_DAgger):
    def __init__(self, env_params=None, reward_function=None, task_id=None, num_tasks=None, fixed_image_size=128, all_tasks=None, task_id_mapping=None):
        super().__init__(env_params=env_params, reward_function=reward_function, task_id=task_id, num_tasks=num_tasks, fixed_image_size=fixed_image_size,
                         all_tasks=all_tasks, task_id_mapping=task_id_mapping)



