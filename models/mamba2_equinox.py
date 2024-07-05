import jax
import jax.random
import jax.numpy as jnp
from jax import jit, vmap, Array

import equinox as eqx
import equinox.nn as nn
from equinox import filter_jit

from einops import rearrange, einsum
from dataclasses import dataclass
from typing import List, Tuple


def truncated_normal_initializer(key, shape):
    assert len(shape) >= 2
    return jax.random.truncated_normal(key, -2, 2, shape) / jnp.sqrt(shape[-2]) * 0.87962566103423978


@dataclass(frozen=True)
class Mamba2Args:
    d_model: int
    n_heads: int 
    n_layers: int
    vocab_size: int
    seq_len: int
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    A_init_range: Tuple[int, int] = (1, 16)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'd_inner', self.expand * self.d_model)
        assert self.d_inner % self.n_heads == 0
        object.__setattr__(self, 'd_head', self.d_inner // self.n_heads)

        object.__setattr__(self, 'd_win', 2 * self.d_inner + 2 * self.d_state + self.n_heads)
        object.__setattr__(self, 'conv_dim', self.d_inner + 2 * self.d_state)


def segsum(x):
    seq_len = x.shape[-1]

    x = jnp.repeat(x[..., None], seq_len, axis=-1) 
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool), k=-1)
    x = jnp.where(mask, x, 0)

    x_segsum = jnp.cumsum(x, axis=-2)
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool), k=0)
    x_segsum = jnp.where(mask, x_segsum, -jnp.inf)
    
    return x_segsum


class Mamba2Layer(eqx.Module):
    norm_in: nn.RMSNorm
    norm_out: nn.RMSNorm
    win: nn.Linear
    conv: Array
    conv_bias: Array | None 
    dt_bias: Array
    A_log: Array
    D: Array
    wout: nn.Linear
    args: Mamba2Args

    def __init__(self, key, args):
        win_key, wout_key, conv_key, dt_key, A_key = jax.random.split(key, 5)
        self.args = args

        A_min, A_max = args.A_init_range
        assert A_min > 0 and A_max >= A_min
        
        self.norm_in = nn.RMSNorm(args.d_model)
        self.norm_out = nn.RMSNorm(args.d_inner)

        self.win = nn.Linear(args.d_model, args.d_win, use_bias=args.bias, key=win_key)
        self.wout = nn.Linear(args.d_inner, args.d_model, use_bias=args.bias, key=wout_key)
        
        self.conv = truncated_normal_initializer(conv_key, (args.conv_dim, args.d_conv))
        self.conv_bias = jnp.zeros(args.conv_dim) if args.conv_bias else None
        
        dt = jax.random.uniform(dt_key, (args.n_heads,))
        dt = jnp.exp(dt * (jnp.log(args.dt_max) - jnp.log(args.dt_min)) + jnp.log(args.dt_min))
        dt = dt.clip(min=args.dt_init_floor)
        self.dt_bias = dt + jnp.log(-jnp.expm1(-dt))
        
        self.A_log = jnp.log(jax.random.uniform(A_key, (args.n_heads,), minval=A_min, maxval=A_max))
        self.D = jnp.ones((args.n_heads,))

    def __call__(self, x):
        x = vmap(self.norm_in)(x)

        zxBCdt = vmap(self.win)(x)

        z, xBC, dt = jnp.split(zxBCdt, [self.args.d_inner, self.args.d_inner + self.args.conv_dim], -1)

        xBC = jnp.concatenate([jnp.zeros((self.args.d_conv - 1, self.args.conv_dim)), xBC], 0).T
        xBC = vmap(jnp.convolve, (0, 0, None))(xBC, self.conv, 'valid').T 

        if self.conv_bias is not None:
            xBC = xBC + self.conv_bias

        xBC = jax.nn.silu(xBC)
        
        x, B, C = jnp.split(xBC, [self.args.d_inner, self.args.d_inner + self.args.d_state], -1)

        y = self.ssd(dt, x, B, C)
        
        y = vmap(self.norm_out)(y * jax.nn.silu(z))

        return vmap(self.wout)(y)
    
    def ssd(self, dt, x, B, C):
        x = rearrange(x, 'l (nh dh) -> nh l dh', nh=self.args.n_heads)
        dt = jax.nn.softplus(dt + self.dt_bias)
        xdt = einsum(x, dt, 'nh l dh, l nh -> nh l dh')  
        Adt = -jnp.exp(self.A_log) * dt
        
        L = jnp.exp(segsum(Adt.T))
        M = L * (C @ B.T)
        y = M @ xdt + einsum(x, self.D, 'nh l dh, nh -> nh l dh')

        return rearrange(y, 'nh l dh -> l (nh dh)')

    def step(self, x, state=None):

        if state is None:
            state = (
                jnp.zeros((self.args.conv_dim, self.args.d_conv - 1)),
                jnp.zeros((self.args.n_heads, self.args.d_head, self.args.d_state))
            )

        conv_state, ssm_state = state

        x = self.norm_in(x)

        zxBCdt = self.win(x) 

        z, xBC, dt = jnp.split(zxBCdt, [self.args.d_inner, self.args.d_inner + self.args.conv_dim], -1)

        conv_input = jnp.concatenate([conv_state, xBC[:, None]], -1)
        xBC = jnp.vecdot(conv_input, jnp.flip(self.conv, -1))

        if self.conv_bias is not None:
            xBC = xBC + self.conv_bias

        xBC = jax.nn.silu(xBC)

        x, B, C = jnp.split(xBC, [self.args.d_inner, self.args.d_inner + self.args.d_state], -1)

        dt = jax.nn.softplus(dt + self.dt_bias)
        x = rearrange(x, '(nh dh) -> nh dh', nh=self.args.n_heads)

        dA = jnp.exp(-jnp.exp(self.A_log) * dt)
        dBx = einsum(B, x, dt, 'n, nh dh, nh -> nh dh n')

        ssm_state = einsum(dA, ssm_state, 'nh, nh dh n -> nh dh n') + dBx 
        y = ssm_state @ C.T + einsum(x, self.D, 'nh dh, nh -> nh dh')

        y = rearrange(y, 'nh dh -> (nh dh)') 

        y = self.norm_out(y * jax.nn.silu(z))

        return self.wout(y), (conv_input[:, 1:], ssm_state)


class Mamba2(eqx.Module):
    wte: Array
    wout: nn.Linear
    layers: List[Mamba2Layer]
    norm: nn.RMSNorm
    args: Mamba2Args
    
    def __init__(self, key, args):
        wte_key, wout_key, layers_key = jax.random.split(key, 3)
        assert args.d_model > args.seq_len
        self.args = args
        self.wte = truncated_normal_initializer(wte_key, (args.vocab_size, args.d_model - args.seq_len))
        self.layers = [Mamba2Layer(subkey, args) for subkey in jax.random.split(layers_key, args.n_layers)]
        self.norm = nn.RMSNorm(args.d_model)
        self.wout = nn.Linear(args.d_model, args.vocab_size, use_bias=False, key=wout_key)
    
    def __call__(self, tokens):
        seq_len = tokens.shape[0]
        x = jnp.concatenate([self.wte[tokens], jnp.eye(seq_len, self.args.seq_len)], -1)

        for layer in self.layers:
            x = x + layer(x)

        return vmap(self.wout)(vmap(self.norm)(x))
    
    @filter_jit
    def step(self, token, position, state=None):
        if state is None:
            state = [None] * self.args.n_layers
        
        x = jnp.concatenate([self.wte[token], jax.nn.one_hot(position, self.args.seq_len)], -1)

        next_state = []
        for layer, layer_state in zip(self.layers, state):
            h, next_layer_state = layer.step(x, layer_state)
            x = x + h
            next_state.append(next_layer_state)

        return self.wout(self.norm(x)), next_state

    def generate(self, steps, tokens, key=None):
        
        def select_token(logits, key):
            if key is None:
                return logits.argmax(), key
            else:
                key, subkey = jax.random.split(key)
                return jax.random.categorical(subkey, logits), key

        position = jnp.array(0)
        state = None

        for token in tokens:
            logits, state = self.step(token, position, state)
            position = position + 1
        
        token, key = select_token(logits, key)
        yield token 
        
        for _ in range(steps - 1):
            logits, state = self.step(token, position, state)
            position = position + 1
            token, key = select_token(logits, key)
            yield token

