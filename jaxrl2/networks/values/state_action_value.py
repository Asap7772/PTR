from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

import jax

from jaxrl2.networks.mlp import MLP
from jaxrl2.networks.mlp import MLPActionSep
from jaxrl2.networks.constants import default_init

from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, jax.lax.Precision, Tuple[str, str],
                      Tuple[jax.lax.Precision, jax.lax.Precision]]

default_kernel_init = nn.initializers.lecun_normal()



class OptionalStopGradientDense(nn.Module):
    """
    A layer to implement optional weights.
    """
    features: int
    use_bias: bool = True
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs, stop_gradient: bool = False):
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param('kernel',
                            self.kernel_init,
                            (inputs.shape[-1], self.features),
                            self.param_dtype)
        kernel = jnp.asarray(kernel, self.dtype)
        kernel = kernel / jnp.sqrt(jnp.sum(kernel ** 2) + 1e-6)
        if stop_gradient:
            kernel = jax.lax.stop_gradient(kernel)

        y = jax.lax.dot_general(inputs, kernel,
                            (((inputs.ndim - 1,), (0,)), ((), ())),
                            precision=self.precision)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,),
                                self.param_dtype)
            bias = jnp.asarray(bias, self.dtype)
            if stop_gradient:
                bias = jax.lax.stop_gradient(bias)
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y



class AuxStateActionValue(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_action_sep: bool = False
    use_pixel_sep: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray, 
                 actions: jnp.ndarray,
                 training: bool=False,
                 version_type: int = 0):
        print ('version type in aux s, a net:', version_type)
        inputs = {'states': observations, 'actions': actions}
        feat = MLP(self.hidden_dims,
                     activations=self.activations)(inputs, training=training)

        # Last layer is special, so handling it here
        critic = nn.Dense(1, kernel_init=default_init(scale=1e-2))(feat)
        adv_layer = OptionalStopGradientDense(1, kernel_init=default_init(scale=1e-2), use_bias=False)

        v1_critic = adv_layer(jax.lax.stop_gradient(feat), stop_gradient=False)
        v2_critic = adv_layer(feat, stop_gradient=True)

        if version_type == 1:
            return jnp.squeeze(v1_critic, -1)
        elif version_type == 2:
            return jnp.squeeze(v2_critic, -1)
        else:
            return jnp.squeeze(critic, -1)



class StateActionValue(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_action_sep: bool = False
    use_normalized_features: bool = False
    use_pixel_sep: bool = False

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 training: bool = False):
        inputs = {'states': observations, 'actions': actions}
        print ('Use action sep in StateActionvalue: ', self.use_action_sep)
        print ('Use normalized features in state action value: ', self.use_normalized_features)
        if self.use_action_sep:
            critic = MLPActionSep(
                (*self.hidden_dims, 1),
                activations=self.activations,
                use_normalized_features=self.use_normalized_features,
                use_pixel_sep=self.use_pixel_sep)(inputs, training=training)
        else:
            critic = MLP((*self.hidden_dims, 1),
                        activations=self.activations,
                        use_normalized_features=self.use_normalized_features)(inputs, training=training)
        return jnp.squeeze(critic, -1)

def cont2disc(values: jnp.ndarray, n: int) -> jnp.ndarray:
    values = (values + 1) / 2
    values = values * n
    values = jnp.floor(values)
    return jnp.clip(values, 0, n - 1)


def disc2cont(values: jnp.ndarray, n: int) -> jnp.ndarray:
    values = values + 0.5
    values = values / n
    return values * 2 - 1


def quantize(values: jnp.ndarray, n: int) -> jnp.ndarray:
    values = cont2disc(values, n)
    return disc2cont(values, n)
    
class StateActionValueAutoregressiveV1(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_action_sep: bool = False
    use_pixel_sep: bool = False
    use_normalized_features: bool = False
    num_components: int = 100
    generate_weights: bool = False

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 training: bool = False,
                 compute_lse: bool = False
                 ):
        print ('Use action sep in StateActionvalue: ', self.use_action_sep)
        print ('Use normalized features in state action value: ', self.use_normalized_features)
        
        actions = cont2disc(actions, self.num_components)
        
        if self.generate_weights:
            # observation conditioned weights for each action dimension
            inputs = {'states': observations}
            weights = MLPActionSep(
                (*self.hidden_dims, actions.shape[-1]),
                activations=self.activations,
                use_normalized_features=self.use_normalized_features,
                use_pixel_sep=self.use_pixel_sep)(inputs, training=training)
        else:
            weights = jnp.ones_like(actions)
        
        # normalize weights
        weights = jnp.exp(weights)/jnp.sum(jnp.exp(weights), axis=-1, keepdims=True)
        
        autoregressive_mlps = []
        
        for i in range(actions.shape[-1]):
            if self.use_action_sep:
                curr_mlp = MLPActionSep(
                    (*self.hidden_dims, 1),
                    activations=self.activations,
                    use_normalized_features=self.use_normalized_features,
                    use_pixel_sep=self.use_pixel_sep)
            else:
                curr_mlp = MLP((*self.hidden_dims, 1),
                            activations=self.activations,
                            use_normalized_features=self.use_normalized_features)
            autoregressive_mlps.append(curr_mlp)
        
        if compute_lse:
            # logsumexp for cql loss
            # not sure if I should compute the lse for the entire action or just till the action dimension that is varied
            q_val_so_far = jnp.zeros((actions.shape[0], actions.shape[-1]*self.num_components))
            which_qval = 0
            for i in range(1,actions.shape[-1]):
                action_so_far = actions[:,:i-1]
                for j in range(self.num_components):
                    which_component = jnp.array([[j]]*actions.shape[0]) # batch_size x 1
                    curr_action_so_far = jnp.concatenate([action_so_far, which_component], axis=-1)
                    critic = autoregressive_mlps[i]({'states': observations, 'actions': curr_action_so_far}, training=training)
                    q_val_so_far = q_val_so_far.at[:, which_qval].set(weights[:,i] * jnp.squeeze(critic, -1))
                    which_qval += 1
            return jax.scipy.special.logsumexp(q_val_so_far, axis=-1)
        else:
            q_val_so_far=0
            for i in range(1,actions.shape[-1]):
                action_so_far = actions[:,:i]
                inputs = {'states': observations, 'actions': action_so_far}
                print('action so far: ', action_so_far.shape)
                
                critic =autoregressive_mlps[i](inputs, training=training)
                q_val_so_far += jnp.squeeze(critic, -1) * weights[:,i]
            return q_val_so_far
        
class StateActionValueAutoregressiveV2(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_action_sep: bool = False
    use_pixel_sep: bool = False
    use_normalized_features: bool = False
    num_components: int = 100
    generate_weights: bool = False
    context_vector_size: int = 128
    weight_sharing: bool = False

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 training: bool = False,
                 compute_lse: bool = False
                 ):
        print ('Use action sep in StateActionvalue: ', self.use_action_sep)
        print ('Use normalized features in state action value: ', self.use_normalized_features)
        
        actions = cont2disc(actions, self.num_components)
        context = jnp.zeros((actions.shape[0], self.context_vector_size)) # B x D
        
        autoregressive_mlps = []
        if self.weight_sharing:
            if self.use_action_sep:
                curr_mlp = MLPActionSep(
                    (*self.hidden_dims, self.context_vector_size),
                    activations=self.activations,
                    use_normalized_features=self.use_normalized_features,
                    use_pixel_sep=self.use_pixel_sep)
            else:
                curr_mlp = MLP((*self.hidden_dims, self.context_vector_size),
                            activations=self.activations,
                            use_normalized_features=self.use_normalized_features)
            autoregressive_mlps = [curr_mlp] * actions.shape[-1]
        else:   
            for i in range(actions.shape[-1]):
                if self.use_action_sep:
                    curr_mlp = MLPActionSep(
                        (*self.hidden_dims, self.context_vector_size),
                        activations=self.activations,
                        use_normalized_features=self.use_normalized_features,
                        use_pixel_sep=self.use_pixel_sep)
                else:
                    curr_mlp = MLP((*self.hidden_dims, self.context_vector_size),
                                activations=self.activations,
                                use_normalized_features=self.use_normalized_features)
                autoregressive_mlps.append(curr_mlp)
        
        output_proj = MLP((1,))
        
        if compute_lse:
            q_val_so_far = jnp.zeros((actions.shape[0], actions.shape[-1]*self.num_components))
            which_qval = 0
            for i in range(1,actions.shape[-1]):
                curr_action = actions[:,i]
                for j in range(self.num_components):
                    which_component = jnp.array([[j]]*actions.shape[0]) # batch_size x 1
                    inputs = {'states': observations, 'actions': which_component, 'context': context}    
                    curr_context = autoregressive_mlps[i](inputs, training=training) + context
                    q_val_so_far = q_val_so_far.at[:, i].set(output_proj(curr_context))
                    which_qval += 1
                
                curr_action = actions[:,i]
                inputs = {'states': observations, 'actions': curr_action, 'context': context}
                context = autoregressive_mlps[i](inputs, training=training) + context
                
            return jax.scipy.special.logsumexp(q_val_so_far, axis=-1)
        else:
            for i in range(1,actions.shape[-1]):
                curr_action = actions[:,i]
                inputs = {'states': observations, 'actions': curr_action, 'context': context}
                
                context = autoregressive_mlps[i](inputs, training=training) + context
            return output_proj(context)