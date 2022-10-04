import ml_collections
from ml_collections.config_dict import config_dict


def get_bc_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 1e-3
    config.hidden_dims = (256, 256)
    config.cosine_decay = True
    config.dropout_rate = 0.1
    config.weight_decay = config_dict.placeholder(float)

    config.distr = 'unitstd_normal'
    # unitstd_normal | tanh_normal | ar_mog

    return config


def get_iql_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.expectile = 0.9  # The actual tau for expectiles.
    config.A_scaling = 10.0
    config.dropout_rate = config_dict.placeholder(float)
    config.cosine_decay = True

    config.tau = 0.005  # For soft target updates.

    config.critic_reduction = 'min'

    config.reward_shift = -1
    config.reward_scale = 1

    return config


def get_config(config_string):
    possible_structures = {
        'bc':
        ml_collections.ConfigDict({
            'model_constructor': 'BCLearner',
            'model_config': get_bc_config()
        }),
        'iql':
        ml_collections.ConfigDict({
            'model_constructor': 'IQLLearner',
            'model_config': get_iql_config()
        })
    }
    return possible_structures[config_string]
