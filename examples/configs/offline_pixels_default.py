import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.value_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.cnn_features = (16, 32, 64, 128, 256)
    config.cnn_strides = (2, 2, 2, 2, 2)
    config.cnn_padding = 'VALID'
    config.latent_dim = 50

    config.discount = 0.96  # corresponds to horizon 25

    config.expectile = 0.7  # The actual tau for expectiles.
    config.A_scaling = 1.0
    config.cosine_decay = True

    config.tau = 0.005

    config.critic_reduction = 'mean'

    config.encoder_type = 'small'
    config.policy_type = 'unit_std_normal'

    config.share_encoders = True

    return config