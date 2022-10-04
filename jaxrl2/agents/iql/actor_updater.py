from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.types import Params, PRNGKey


def update_actor(key: PRNGKey, actor: TrainState, target_critic: TrainState,
                 value: TrainState, batch: FrozenDict, A_scaling: float,
                 critic_reduction: str,
                 ) -> Tuple[TrainState, Dict[str, float]]:

    if value.batch_stats is not None:
        v = value.apply_fn({'params': value.params, 'batch_stats': value.batch_stats}, batch['observations'],
                           training=False, mutable=False)
    else:
        v = value.apply_fn({'params': value.params}, batch['observations'])

    if target_critic.batch_stats is not None:
        qs = target_critic.apply_fn({'params': target_critic.params, 'batch_stats': target_critic.batch_stats},
                                    batch['observations'], batch['actions'], training=False, mutable=False)
    else:
        qs = target_critic.apply_fn({'params': target_critic.params},
                                    batch['observations'], batch['actions'])
    if critic_reduction == 'min':
        q = qs.min(axis=0)
    elif critic_reduction == 'mean':
        q = qs.mean(axis=0)
    else:
        raise NotImplemented()
    exp_a = jnp.exp((q - v) * A_scaling)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(
            actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        if actor.batch_stats is not None:
            dist, new_model_state = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats},
                                  batch['observations'],
                                  training=True,
                                  mutable=['batch_stats'],
                                  rngs={'dropout': key})
        else:
            dist = actor.apply_fn({'params': actor_params},
                                  batch['observations'],
                                  training=True,
                                  rngs={'dropout': key})
            new_model_state = {}
        log_probs = dist.log_prob(batch['actions'])
        actor_loss = -(exp_a * log_probs).mean()

        mse = (dist.loc - batch['actions']) ** 2
        mse = mse.mean()

        infos = {'actor_loss': actor_loss, 'adv': q - v, 'weights': exp_a, 'mse': mse, 'pred_actions_mean': dist.loc, 'dataset_actions': batch['actions'],
                 'action_std': dist._scale_diag}
        return actor_loss, (new_model_state, infos)

    grads, (new_model_state, info) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info
