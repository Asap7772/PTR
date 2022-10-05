from jaxrl2.agents import PixelBCLearner
from jaxrl2.agents.sarsa import PixelSARSALearner
from jaxrl2.agents.cql_encodersep_parallel.pixel_cql_learner import PixelCQLLearnerEncoderSepParallel
from jaxrl2.agents.cql_encodersep_parallel_awbc.pixel_cql_learner import PixelAWBCLearnerEncoderSepParallel
from jaxrl2.agents import PixelBCLearner

def get_algo(variant, sample_obs, sample_action, **kwargs):
    if variant.algorithm == 'bc':
        agent = PixelBCLearner(variant.seed, sample_obs, sample_action, **kwargs)
    elif variant.algorithm == 'awbc':
        agent = PixelAWBCLearnerEncoderSepParallel(variant.seed, sample_obs, 
                                    sample_action, **kwargs)
    elif variant.algorithm == 'sarsa':
        agent = PixelSARSALearner(variant.seed, sample_obs, 
                                sample_action, **kwargs)
    elif variant.algorithm == 'cql_encodersep_parallel':
        agent = PixelCQLLearnerEncoderSepParallel(variant.seed, sample_obs, sample_action, **kwargs)
    else:
        assert False, 'unknown algorithm'
    return agent