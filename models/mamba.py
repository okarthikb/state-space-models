import math
import jax
import jax.nn as nn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as random
from jax import jit, vmap, Array
from einops import repeat, einsum
from functools import partial
from dataclasses import dataclass
from typing import Tuple, NamedTuple


@dataclass
class ModelArgs:
    d_model: int
    n_layers: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: int | str = 'auto'
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_scale = 1.0
    dt_init_floor = 1e-4
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        
        self.orig_vocab_size = self.vocab_size

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple


class LayerParams(NamedTuple):
    norm: Array
    in_proj: Array
    in_proj_bias: Array | None
    conv: Array
    conv_bias: Array | None
    x_proj: Array
    dt_proj: Array
    dt_proj_bias: Array
    A_log: Array
    D: Array
    out_proj: Array
    out_proj_bias: Array | None


class MambaParams(NamedTuple):
    embedding: Array
    layers: LayerParams
    norm_f: Array


def initialize_params(key, args):
    truncated_normal_stddev = .87962566103423978

    d_model_scale = 1 / (math.sqrt(args.d_model) * truncated_normal_stddev)
    d_inner_scale = 1 / (math.sqrt(args.d_inner) * truncated_normal_stddev)
    dt_rank_scale = 1 / (math.sqrt(args.dt_rank) * truncated_normal_stddev)
    dt_init_std = args.dt_rank ** -0.5 * args.dt_scale

    embed_key, dt_key, layers_key = random.split(key, 3)
    layers_keys = random.split(layers_key, 6)

    embedding = random.truncated_normal(embed_key, -2, 2, (args.vocab_size, args.d_model)) * d_model_scale
    
    norm = jnp.ones((args.n_layers, args.d_model))
    
    in_proj = random.truncated_normal(
        layers_keys[0], -2, 2, (args.n_layers, args.d_model, args.d_inner * 2)
    ) * d_model_scale
    in_proj_bias = jnp.zeros((args.n_layers, args.d_inner)) if args.bias else None

    conv = random.truncated_normal(
        layers_keys[1], -2, 2, (args.n_layers, args.d_inner, args.d_conv)
    ) * d_inner_scale
    conv_bias = jnp.zeros((args.n_layers, args.d_inner)) if args.conv_bias else None
    
    x_proj = random.truncated_normal(
        layers_keys[2], -2, 2, (args.n_layers, args.d_inner, args.dt_rank + 2 * args.d_state)
    ) * d_inner_scale

    dt_proj = random.uniform(
        layers_keys[3],
        (args.n_layers, args.dt_rank, args.d_inner),
        minval=-dt_init_std,
        maxval=dt_init_std
    )
    dt = random.uniform(layers_keys[4], (args.n_layers, args.d_inner))
    dt = jnp.exp(dt * (math.log(args.dt_max) - math.log(args.dt_min)) + math.log(args.dt_min))
    dt = dt.clip(min=args.dt_init_floor)
    dt_proj_bias = dt + jnp.log(-jnp.expm1(-dt))
    
    A_log = repeat(
        jnp.log(jnp.arange(1, args.d_state + 1)), 'ds -> nl di ds',
        nl=args.n_layers,
        di=args.d_inner
    )
    D=jnp.ones((args.n_layers, args.d_inner))
    
    out_proj = random.truncated_normal(
        layers_keys[5], -2, 2, (args.n_layers, args.d_inner, args.d_model)
    ) * d_inner_scale
    out_proj_bias=jnp.zeros((args.n_layers, args.d_model)) if args.bias else None

    layers = LayerParams(
        norm=norm,
        in_proj=in_proj,
        in_proj_bias=in_proj_bias,
        conv=conv,
        conv_bias=conv_bias,
        x_proj=x_proj,
        dt_proj=dt_proj,
        dt_proj_bias=dt_proj_bias,
        A_log=A_log,
        D=D,
        out_proj=out_proj,
        out_proj_bias=out_proj_bias,
    )

    norm_f = jnp.ones(args.d_model)

    return MambaParams(embedding=embedding, layers=layers, norm_f=norm_f)


def zero_or(x):
    return 0 if x is None else x


def rms_norm(w, x, eps):
    z = x.astype(jnp.float32)
    norm = z * lax.rsqrt(jnp.mean(z * z, -1, keepdims=True) + eps)
    return w * norm.astype(x.dtype)


# training
def mamba(args, use_associative_scan, params, tokens):

    def block(x, params):
        # (seq_len, d_model) -> (seq_len, d_model * 2) -> (seq_len, d_model), (seq_len, d_model)
        x, res = jnp.split(x @ params.in_proj + zero_or(params.in_proj_bias), 2, -1)

        # (seq_len, d_inner) -> (seq_len + d_conv - 1, d_inner) -> (d_inner, seq_len + d_conv - 1)
        x = jnp.concatenate([jnp.zeros((args.d_conv - 1, args.d_inner)), x], 0).T
        # (d_model, seq_len + d_conv - 1) -> (d_model, seq_len) -> (seq_len, d_model)
        x = vmap(jnp.convolve, (0, 0, None))(x, params.conv, 'valid').T + zero_or(params.conv_bias)
        x = nn.silu(x)

        # (seq_len, d_inner) -> (seq_len, dt_rank), (seq_len, d_state), (seq_len, d_state)
        x_dt, B, C = jnp.split(x @ params.x_proj, [args.dt_rank, args.dt_rank + args.d_state], -1)

        # (seq_len, dt_rank) -> (seq_len, d_inner)
        dt = nn.softplus(x_dt @ params.dt_proj + zero_or(params.dt_proj_bias))

        # discretization
        dA = jnp.exp(einsum(dt, -jnp.exp(params.A_log), 'l d, d s -> l d s'))
        dBx = einsum(x * dt, B, 'l d, l s -> l d s')

        # see section 1.4.1 "First-Order Recurrences" in the paper "Prefix Sums and Their Applications"
        # the main loop is equivalent to
        # 
        # ssm_states = []
        # s = jnp.zeros((args.d_inner, args.d_state))
        # for c in zip(dA, dBx):
        #     s = c[0] * s + c[1]
        #     ssm_states.append(s)
        # ssm_states = jnp.stack(ssm_states)
        #
        # we use the associative operator `op` below to parallelize this
        if use_associative_scan:
            op = lambda s, c: (c[0] * s[0], c[0] * s[1] + c[1])
            _, ssm_states = lax.associative_scan(op, (dA, dBx))
        # or we can implement the same loop using lax.scan 
        else:
            def op(s, c):
                s = c[0] * s + c[1]
                return s, s

            _, ssm_states = lax.scan(op, jnp.zeros((args.d_inner, args.d_state)), (dA, dBx))

        # read out, gating, then output projection
        y = einsum(ssm_states, C, 'l d s, l s -> l d') + x * params.D
        y = y * nn.silu(res)

        # (seq_len, d_inner) -> (seq_len, d_model)
        return y @ params.out_proj + zero_or(params.out_proj_bias)

    def f(x, params):
        return x + block(rms_norm(params.norm, x, 1e-8), params), None

    h, _ = lax.scan(f, params.embedding[tokens], params.layers)
    
    logits = rms_norm(params.norm_f, h, 1e-8) @ params.embedding.T

    return logits


def mamba_step(args, valid_logits, params, token, cache=None):

    def block(x, params, conv_cache, ssm_state):
        # (d_model,) -> (d_inner,), (d_inner,)
        x, res = jnp.split(x @ params.in_proj + zero_or(params.in_proj_bias), 2, -1)

        # conv step
        conv_input = jnp.concatenate([conv_cache, x[:, None]], -1)  # (d_inner, d_conv)
        kernel = jnp.flip(params.conv, -1)  # (d_inner, d_conv)
        x = nn.silu(jnp.vecdot(conv_input, kernel) + zero_or(params.conv_bias))

        x_dt, B, C = jnp.split(x @ params.x_proj, [args.dt_rank, args.dt_rank + args.d_state], -1)

        dt = nn.softplus(x_dt @ params.dt_proj + zero_or(params.dt_proj_bias))

        # discretization
        dA = jnp.exp(einsum(-jnp.exp(params.A_log), dt, 'd n, d -> d n'))
        dBx = einsum(B, x * dt, 'n, d -> d n')

        # SSM step
        ssm_state = dA * ssm_state + dBx
        y = ssm_state @ C.T + x * params.D

        y = y * nn.silu(res)

        y = y @ params.out_proj + zero_or(params.out_proj_bias)

        return y, (conv_input[:, 1:], ssm_state)

    def f(x, params_and_cache):
        params, cache = params_and_cache
        h, cache = block(rms_norm(params.norm, x, 1e-8), params, *cache)
        return x + h, cache

    if cache is None:
        cache = (
            jnp.zeros((args.n_layers, args.d_inner, args.d_conv - 1)),
            jnp.zeros((args.n_layers, args.d_inner, args.d_state))
        )

    x, cache = lax.scan(f, params.embedding[token], (params.layers, cache))

    logits = rms_norm(params.norm_f, x, 1e-8) @ params.embedding.T
    
    return logits[:args.orig_vocab_size if valid_logits else args.vocab_size], cache


def generate(key, args, params, tokenizer, steps, temperature, prompt, cache=None):
    print(prompt, end='')
    
    f = jit(partial(mamba_step, args, True, params))

    tokens = jnp.array(tokenizer.encode(prompt), dtype=jnp.int32)

    for token in tokens:
        logits, cache = f(token, cache)
  
    token = random.categorical(key, logits / temperature)
    print(tokenizer.id_to_token(int(token)), end='')

    for _ in range(steps):
        key, subkey = random.split(key)
        logits, cache = f(token, cache)
        token = random.categorical(subkey, logits / temperature)
        print(tokenizer.id_to_token(int(token)), end='')
    print()
    
    return cache

