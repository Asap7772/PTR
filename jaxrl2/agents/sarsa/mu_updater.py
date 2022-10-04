from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey

def update_mu(key: PRNGKey, actor: TrainState, mu_actor: TrainState, critic: TrainState, temp: float, batch: DatasetDict, neg_adv:bool=False) -> Tuple[TrainState, Dict[str, float]]:
    def actor_loss_fn(mu_actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        # use actual actor
        dist = actor.apply_fn({'params': actor.params}, batch['observations'])
        actions, log_pis = dist.sample_and_log_prob(seed=key)

        # use mu_actor
        mu_dist = mu_actor.apply_fn({'params': mu_actor_params}, batch['observations'])
        mu_log_probs = mu_dist.log_prob(batch['actions'])

        qs = critic.apply_fn({'params': critic.params}, batch['observations'], batch['actions'])
        vs = critic.apply_fn({'params': critic.params}, batch['observations'], actions)

        advantage = (qs - vs.min(0)).mean(0)
        advantage = advantage.reshape(mu_log_probs.shape) # shape: [batch_size]

        if neg_adv:
            advantage = -advantage

        weight = jnp.clip(advantage*temp, a_min=-10, a_max=5) # clip advantage to [-10, 20]

        actor_loss = -(mu_log_probs * jnp.exp(weight))

        things_to_log = {
            'actor_loss': actor_loss.mean(),
            'actor_loss_std': actor_loss.std(),

            'log_probs': mu_log_probs.mean(),
            'log_probs_std': mu_log_probs.std(),
            'log_pis': log_pis.mean(),
            'log_pis_std': log_pis.std(),

            'entropy': -log_pis.mean(),
            'q_pi_in_actor': qs.mean(),
            
            'advantage_mean': advantage.mean(),
            'advantage_std': advantage.std(),
            'advantage_max': advantage.max(),
            'advantage_min': advantage.min(),

            'weight_mean': weight.mean(),
            'weight_std': weight.std(),
            'weight_max': weight.max(),
            'weight_min': weight.min(),
        }

        return actor_loss.mean(), things_to_log

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(mu_actor.params)
    new_actor = mu_actor.apply_gradients(grads=grads)

    return new_actor, info
