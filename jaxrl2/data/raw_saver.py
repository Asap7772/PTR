from widowx_envs.utils.datautils.raw_saver import RawSaver
import numpy as np

class RawSaverJaxRL(RawSaver):
    def __init__(self, save_dir, unnormalize_func=None):
        super(RawSaverJaxRL, self).__init__(save_dir, ngroup=1000)
        self.i_traj = 0
        self.unnormalize_func = unnormalize_func

    def save(self, path):
        path = convert_listofdicts2dictoflists(path)

        obs = path['full_observation']
        obs.append(path['next_full_observation'][-1])
        obs = convert_listofdicts2dictoflists(obs)

        if self.unnormalize_func is not None:
            actions = [self.unnormalize_func(act) for act in path['action']]

        self.save_traj(self.i_traj, {}, obs, actions, path['reward'])
        print('saving traj {} done'.format(self.i_traj))
        self.i_traj += 1


def convert_listofdicts2dictoflists(list):
    obs_dict = {}
    for key in list[0].keys():
        elements = []
        for tstep in list:
            elements.append(tstep[key])
        if isinstance(elements[0], np.ndarray):
            elements = np.stack(elements, 0)
        obs_dict[key] = elements
    return obs_dict
