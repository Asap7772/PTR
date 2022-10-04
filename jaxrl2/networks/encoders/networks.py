from typing import Dict, Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init, xavier_init, kaiming_init

from functools import partial
from typing import Any, Callable, Sequence, Tuple
import distrax
from jaxrl2.networks.values import StateActionEnsemble, StateValue, AuxStateActionEnsemble, StateActionEnsembleAutoregressive

ModuleDef = Any

class Encoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, observations: jnp.ndarray, training=False) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        for features, stride in zip(self.features, self.strides):
            x = nn.Conv(features,
                        kernel_size=(3, 3),
                        strides=(stride, stride),
                        kernel_init=default_init(),
                        padding=self.padding)(x)
            x = nn.relu(x)

        return x.reshape((*x.shape[:-3], -1))
class IdentityEncoder(nn.Module):
    @nn.compact
    def __call__(self, observations: jnp.ndarray, training=False) -> jnp.ndarray:
        return observations.reshape(observations.shape[0], -1) # flatten

class LateFusionEncoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        num_encoders = observations.shape[-1]
        embeddings = []
        for i in range(num_encoders):
            embeddings.append(Encoder(self.features, self.strides, self.padding)(observations[..., i][..., None]))
        return jnp.concatenate(embeddings, axis=-1)


class PixelMultiplexer(nn.Module):
    encoder: Union[nn.Module, list]
    network: nn.Module
    latent_dim: int
    stop_gradient: bool = False
    use_bottleneck: bool=True
    use_multiplicative_cond: bool = False

    @nn.compact
    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 actions: Optional[jnp.ndarray] = None,
                 training: bool = False):
        
        observations = FrozenDict(observations)

        if self.use_multiplicative_cond:
            x = self.encoder(observations['pixels'], training,
                             cond_var=observations['task_id'])
        else:
            x = self.encoder(observations['pixels'], training)

        if self.stop_gradient:
            # We do not update conv layers with policy gradients.
            x = jax.lax.stop_gradient(x)

        if self.use_bottleneck:
            x = nn.Dense(self.latent_dim, kernel_init=xavier_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        x = observations.copy(add_or_replace={'pixels': x})

        print('fully connected keys', x.keys())
        if actions is None:
            return self.network(x, training=training)
        else:
            return self.network(x, actions, training=training)
        

'''
Split into Encoder and Decoder
'''
class PixelMultiplexerEncoder(nn.Module):
    encoder: Union[nn.Module, list]
    latent_dim: int
    stop_gradient: bool = False
    use_bottleneck: bool=True
    use_multiplicative_cond: bool = False

    @nn.compact
    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 training: bool = False):
        observations = FrozenDict(observations)

        if self.use_multiplicative_cond:
            x = self.encoder(observations['pixels'], training, cond_var=observations['task_id'])
        else:
            x = self.encoder(observations['pixels'], training)

        if self.stop_gradient:
            # We do not update conv layers with policy gradients.
            x = jax.lax.stop_gradient(x)

        if self.use_bottleneck:
            x = nn.Dense(self.latent_dim, kernel_init=xavier_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
            # x = nn.relu(x)

        return observations.copy(add_or_replace={'pixels': x})


class PixelMultiplexerEncoderWithoutFinal(nn.Module):
    encoder: Union[nn.Module, list]
    latent_dim: int
    stop_gradient: bool = False
    use_bottleneck: bool=False
    use_multiplicative_cond: bool = False

    @nn.compact
    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 training: bool = False):
        observations = FrozenDict(observations)

        if self.use_multiplicative_cond:
            x = self.encoder(observations['pixels'], training, cond_var=observations['task_id'])
        else:
            x = self.encoder(observations['pixels'], training)
        
        if self.use_bottleneck:
            x = nn.Dense(self.latent_dim, kernel_init=xavier_init())(x)

        print ('Using encoder without final.... but will add dropout')
        return observations.copy(add_or_replace={'pixels': x})


class PixelMultiplexerDecoderWithDropout(nn.Module):
    network: nn.Module
    dropout_rate: Optional[float] = None
    
    @nn.compact
    def __call__(self, 
                 embedding: Union[FrozenDict, Dict], 
                 actions: Optional[jnp.ndarray] = None, 
                 training: bool = False,
                 compute_lse: bool = False):
        
        if self.dropout_rate is not None:
            print ('Pixel multiplexer decoder with dropout.... using dataset')
            # Now apply dropout on the image embedding to prevent it
            # from overfitting, this is done here, because if we do it
            # before, we will only have the state embedding with the same
            # dropout mask...
            x = embedding['pixels']
            x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
            embedding = embedding.copy(add_or_replace={'pixels': x})
            
        if isinstance(self.network, StateActionEnsembleAutoregressive):
            if actions is None:
                return self.network(embedding, training=training, compute_lse=compute_lse)
            else:
                return self.network(embedding, actions, training=training, compute_lse=compute_lse)
        else:
            if actions is None:
                return self.network(embedding, training=training)
            else:
                return self.network(embedding, actions, training=training)


class PixelMultiplexerDecoder(nn.Module):
    network: nn.Module
    
    @nn.compact
    def __call__(self, 
                 embedding: Union[FrozenDict, Dict], 
                 actions: Optional[jnp.ndarray] = None, 
                 training: bool = False,
                 compute_lse: bool = False):
        
        if isinstance(self.network, StateActionEnsembleAutoregressive):
            if actions is None:
                return self.network(embedding, training=training, compute_lse=compute_lse)
            else:
                return self.network(embedding, actions, training=training, compute_lse=compute_lse)
        else:
            if actions is None:
                return self.network(embedding, training=training)
            else:
                return self.network(embedding, actions, training=training)


class AuxPixelMultiplexerDecoder(nn.Module):
    network: nn.Module
    
    @nn.compact
    def __call__(self, 
                 embedding: Union[FrozenDict, Dict], 
                 actions: Optional[jnp.ndarray] = None, 
                 training: bool = False,
                 version_type: int = 0, 
                 compute_lse: bool = False):
        if isinstance(self.network, StateActionEnsembleAutoregressive):
            if actions is None:
                return self.network(embedding, training=training, version_type=version_type, compute_lse=compute_lse)
            else:
                return self.network(embedding, actions, training=training, version_type=version_type, compute_lse=compute_lse)
        else:
            if actions is None:
                return self.network(embedding, training=training)
            else:
                return self.network(embedding, actions, training=training)
