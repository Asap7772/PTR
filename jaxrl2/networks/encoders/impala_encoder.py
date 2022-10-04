import flax.linen as nn
import jax.numpy as jnp


from jaxrl2.networks.constants import default_init, xavier_init, kaiming_init


class ResnetStack(nn.Module):
    num_ch: int
    num_blocks: int
    use_max_pooling: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_ch,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME'
        )(observations)

        if self.use_max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2)
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_ch, kernel_size=(3, 3), strides=1,
                padding='SAME',
                kernel_init=initializer)(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_ch, kernel_size=(3, 3), strides=1,
                padding='SAME', kernel_init=initializer
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    nn_scale: int = 1
    use_multiplicative_cond: bool = False

    def setup(self):
        stack_sizes = [16, 32, 32]
        self.stack_blocks = [
            ResnetStack(
                num_ch=stack_sizes[0] * self.nn_scale,
                num_blocks=2),
            ResnetStack(
                num_ch=stack_sizes[1] * self.nn_scale,
                num_blocks=2),
            ResnetStack(
                num_ch=stack_sizes[2] * self.nn_scale,
                num_blocks=2),
        ]

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.use_multiplicative_cond:
                assert cond_var is not None, "Cond var shouldn't be done when using it"
                print("Using Multiplicative Cond!")
                temp_out = nn.Dense(conv_out.shape[-1], kernel_init=xavier_init())(cond_var)
                x_mult = jnp.expand_dims(jnp.expand_dims(temp_out, 1), 1)
                print ('x_mult shape in IMPALA:', x_mult.shape, conv_out.shape)
                conv_out = conv_out * x_mult

        conv_out = nn.relu(conv_out)
        return conv_out.reshape((*x.shape[:-3], -1))
    

class BiggerImpalaEncoder(nn.Module):
    nn_scale: int = 1
    use_multiplicative_cond: bool = False

    def setup(self):
        stack_sizes = [16, 32, 32, 32]
        self.stack_blocks = [
            ResnetStack(
                num_ch=stack_sizes[0] * self.nn_scale,
                num_blocks=2),
            ResnetStack(
                num_ch=stack_sizes[1] * self.nn_scale,
                num_blocks=2),
            ResnetStack(
                num_ch=stack_sizes[2] * self.nn_scale,
                num_blocks=2),
            ResnetStack(
                num_ch=stack_sizes[2] * self.nn_scale,
                num_blocks=2),
        ]

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        conv_out = x
        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.use_multiplicative_cond:
                assert cond_var is not None, "Cond var shouldn't be done when using it"
                print("Using Multiplicative Cond!")
                temp_out = nn.Dense(conv_out.shape[-1], kernel_init=xavier_init())(cond_var)
                x_mult = jnp.expand_dims(jnp.expand_dims(temp_out, 1), 1)
                print ('x_mult shape in IMPALA:', x_mult.shape, conv_out.shape)
                conv_out = conv_out * x_mult

        conv_out = nn.relu(conv_out)
        return conv_out.reshape((*x.shape[:-3], -1))
    
    
class BiggestImpalaEncoder(nn.Module):
    nn_scale: int = 1
    use_multiplicative_cond: bool = False

    def setup(self):
        stack_sizes = [16, 32, 32, 32, 32]
        self.stack_blocks = [
            ResnetStack(
                num_ch=stack_sizes[0] * self.nn_scale,
                num_blocks=2),
            ResnetStack(
                num_ch=stack_sizes[1] * self.nn_scale,
                num_blocks=2),
            ResnetStack(
                num_ch=stack_sizes[2] * self.nn_scale,
                num_blocks=2),
            ResnetStack(
                num_ch=stack_sizes[2] * self.nn_scale,
                num_blocks=2),
            ResnetStack(
                num_ch=stack_sizes[2] * self.nn_scale,
                num_blocks=2),
        ]

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.use_multiplicative_cond:
                assert cond_var is not None, "Cond var shouldn't be done when using it"
                print("Using Multiplicative Cond!")
                temp_out = nn.Dense(conv_out.shape[-1], kernel_init=xavier_init())(cond_var)
                x_mult = jnp.expand_dims(jnp.expand_dims(temp_out, 1), 1)
                print ('x_mult shape in IMPALA:', x_mult.shape, conv_out.shape)
                conv_out = conv_out * x_mult

        conv_out = nn.relu(conv_out)
        return conv_out.reshape((*x.shape[:-3], -1))


class SmallerImpalaEncoder(nn.Module):
    nn_scale: int = 1

    def setup(self):
        stack_sizes = [16, 32, 32]
        self.stack_blocks = [
            ResnetStack(
                num_ch=stack_sizes[0] * self.nn_scale,
                num_blocks=2),
            ResnetStack(
                num_ch=stack_sizes[1] * self.nn_scale,
                num_blocks=1),
            ResnetStack(
                num_ch=stack_sizes[2] * self.nn_scale,
                num_blocks=1),
        ]

    @nn.compact
    def __call__(self, x, train=True):
        x = x.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)

        conv_out = nn.relu(conv_out)
        return conv_out.reshape((*x.shape[:-3], -1))
