from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.values.state_action_value import StateActionValue, StateActionValueAutoregressiveV1, StateActionValueAutoregressiveV2
from jaxrl2.networks.values.state_action_value import AuxStateActionValue


class StateActionEnsemble(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2
    use_action_sep: bool = False
    use_normalized_features: bool = False
    use_pixel_sep: bool = False

    @nn.compact
    def __call__(self, states, actions, training: bool = False):

        print ('Use action sep in state action ensemble: ', self.use_action_sep)
        VmapCritic = nn.vmap(StateActionValue,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_qs)
        qs = VmapCritic(self.hidden_dims,
                        activations=self.activations,
                        use_action_sep=self.use_action_sep,
                        use_normalized_features=self.use_normalized_features,
                        use_pixel_sep=self.use_pixel_sep)(
                            states, actions, training)
        return qs
    

class StateActionEnsembleAutoregressive(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2
    use_action_sep: bool = False
    use_normalized_features: bool = False
    use_pixel_sep: bool = False
    autoregressive_type: int = 0

    @nn.compact
    def __call__(self, states, actions, training: bool = False, compute_lse: bool = False):

        print('Autoregressive type: ', self.autoregressive_type)
        print ('Use action sep in state action ensemble: ', self.use_action_sep)
        if self.autoregressive_type == 0:
            vmap_type = StateActionValueAutoregressiveV1
        elif self.autoregressive_type == 1:
            vmap_type = StateActionValueAutoregressiveV2
        else:
            raise ValueError('Unknown autoregressive type')
        
        VmapCritic = nn.vmap(vmap_type,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_qs)
        
        qs = VmapCritic(self.hidden_dims,
                        activations=self.activations,
                        use_action_sep=self.use_action_sep,
                        use_normalized_features=self.use_normalized_features,
                        use_pixel_sep=self.use_pixel_sep)(
                            states, actions, training, compute_lse)
        return qs


class AuxStateActionEnsemble(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2
    use_action_sep: bool = False
    use_pixel_sep: bool = False

    @nn.compact
    def __call__(self, states, actions, training: bool=False,
                 version_type: int = 0):
        print ('version_type in aux discrete:', version_type)
        # VmapCritic = nn.vmap(AuxDiscreteStateActionValue,
        #                      variable_axes={'params': 0, 'version_type': None},
        #                      split_rngs={'params': True},
        #                      in_axes=None,
        #                      out_axes=0,
        #                      axis_size=self.num_qs)
        # qs = VmapCritic(self.hidden_dims, num_actions=self.num_actions,
        #                 activations=self.activations)(
        #                     states, version_type=version_type)
        outputs = []
        for _ in range(self.num_qs):
            outputs.append(
                AuxStateActionValue(
                    self.hidden_dims, use_action_sep=self.use_action_sep,
                    activations=self.activations,
                    use_pixel_sep=self.use_pixel_sep
                )(states, actions, training=training, version_type=version_type)
            )
        return jnp.stack(outputs, axis=0)
        # return qs