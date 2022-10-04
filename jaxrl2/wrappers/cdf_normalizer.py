import gym
import numpy as np
import scipy.interpolate as interpolate


class CDFNormalizer(gym.ActionWrapper):

    def __init__(self,
                 env: gym.Env,
                 actions: np.ndarray,
                 update_actions_inplace: bool = False,
                 eps: float = 1e-6):
        super().__init__(env)
        assert actions.ndim > 1

        self.dim = actions.shape[-1]
        self.cdfs = [CDFNormalizer1d(actions[..., i]) for i in range(self.dim)]
        if update_actions_inplace:
            actions[:] = self.wrap('normalize', actions)
            lim = 1 - eps
            actions[:] = np.clip(actions, -lim, lim)

    def wrap(self, fn_name, x):
        out = np.empty_like(x)
        for i, cdf in enumerate(self.cdfs):
            fn = getattr(cdf, fn_name)
            out[..., i] = fn(x[..., i])
        return out

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        return self.wrap('normalize', action)

    def action(self, action: np.ndarray) -> np.ndarray:
        return self.wrap('unnormalize', action)


class CDFNormalizer1d(object):

    def __init__(self, action: np.ndarray):
        assert action.ndim == 1
        self.X = action.astype(np.float32)
        quantiles, cumprob = empirical_cdf(self.X)
        self.fn = interpolate.interp1d(quantiles, cumprob)
        self.inv = interpolate.interp1d(cumprob, quantiles)

        self.xmin, self.xmax = quantiles.min(), quantiles.max()
        self.ymin, self.ymax = cumprob.min(), cumprob.max()

    def normalize(self, x):
        x = np.clip(x, self.xmin, self.xmax)
        ## [ 0, 1 ]
        y = self.fn(x)
        ## [ -1, 1 ]
        y = 2 * y - 1
        return y

    def unnormalize(self, x):
        '''
            X : [ -1, 1 ]
        '''
        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.0

        # if (x < self.ymin).any() or (x > self.ymax).any():
        #     print(f'[ dataset/normalization ] Warning: out of range in unnormalize: [{x.min()}, {x.max()}] | [{self.ymin}, {self.ymax}]')
        x = np.clip(x, self.ymin, self.ymax)

        y = self.inv(x)
        return y


def empirical_cdf(sample):
    ## https://stackoverflow.com/a/33346366

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob
