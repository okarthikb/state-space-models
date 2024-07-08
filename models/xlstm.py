# note
#
# xLSTM has an sLSTM and mLSTM component. mLSTM like a few recent recurrent architectures
# has parallel and sequential modes for fast training and inference respectively, which
# is pretty cool, so this file implements only the mLSTM component
#
# original repo https://github.com/NX-AI/xlstm.git
# paper https://arxiv.org/abs/2405.04517


import math

import jax
import jax.random
import jax.numpy as jnp
from jax import vmap

import equinox as eqx
import equinox.nn as nn
from equinox import filter_jit

from einops import rearrange, repeat, einsum

from dataclasses import dataclass

from typing import Tuple, List, Callable
from jaxtyping import Float, Bool, Array, jaxtyped
from beartype import beartype


typecheck = jaxtyped(typechecker=beartype)


pick_weight = lambda linear: linear.weight
pick_bias = lambda linear: linear.bias


def small_init(key, shape, dim):
    return jax.random.normal(key, shape) * math.sqrt(2 / (5 * dim))


def wang_init(key, shape, dim, n_layers):
    return jax.random.normal(key, shape) * 2 / (n_layers * math.sqrt(dim))


def set_linear_weight(key, in_features, out_features, use_bias, init_fn, n_layers=None) -> nn.Linear:
    linear = nn.Linear(in_features, out_features, use_bias=use_bias, key=key)

    if init_fn == 'small':
        weight = small_init(key, (out_features, in_features), in_features)
    elif init_fn == 'wang':
        assert n_layers is not None
        weight = wang_init(key, (out_features, in_features), in_features, n_layers)
    elif init_fn == 'zero':
        weight = jnp.zeros_like(linear.weight)
    else:
        raise ValueError(f"{init_fn} is not a valid")
    
    return eqx.tree_at(pick_weight, linear, weight)


@dataclass(frozen=True)
class xLSTMArgs:
    d_model: int
    n_heads: int
    n_layers: int
    vocab_size: int
    seq_len: int
    d_conv: int = 4
    xlstm_expand: int = 2
    ffn_expand: int = 2
    use_ln_bias: bool = False
    use_proj_bias: bool = True
    use_qkv_bias: bool = False
    use_conv_bias: bool = True
    use_ffn_bias: bool = False
    use_ffn: bool = True
    tie_weights: bool = True
    activation: Callable[[Array], Array] = jax.nn.gelu

    def __post_init__(self):
        object.__setattr__(self, 'd_inner', self.d_model * self.xlstm_expand)
        object.__setattr__(self, 'd_hidden', self.d_model * self.ffn_expand)

        assert self.d_model % self.n_heads == 0 
        object.__setattr__(self, 'd_head', self.d_inner // self.n_heads)


class GatedFFN(eqx.Module):
    win: nn.Linear
    wout: nn.Linear
    d_hidden: int
    activation: Callable[[Array], Array]
    
    def __init__(self, key, d_model, d_hidden, use_ffn_bias, n_layers=1, activation=jax.nn.gelu):
        win_key, wout_key = jax.random.split(key)
        self.d_hidden = d_hidden
        self.activation = activation
        self.win = set_linear_weight(win_key, d_model, 2 * d_hidden, use_ffn_bias, 'small')
        self.wout = set_linear_weight(wout_key, d_hidden, d_model, use_ffn_bias, 'wang', n_layers)
        if use_ffn_bias:
            self.win = eqx.tree_at(pick_bias, self.win, jnp.zeros_like(self.win.bias))
            self.wout = eqx.tree_at(pick_bias, self.wout, jnp.zeros_like(self.wout.bias))
    
    def __call__(self, x):
        x, z = jnp.split(self.win(x), [self.d_hidden], -1)
        return self.wout(self.activation(x) * z)


class HeadLinear(eqx.Module):
    weight: Array
    bias: Array | None = None
    n_heads: int 

    def __init__(self, key, in_features, n_heads, use_bias):
        assert in_features % n_heads == 0
        d_head = in_features // n_heads
        self.n_heads = n_heads
        self.weight = small_init(key, (n_heads, d_head, d_head), d_head) 
        if use_bias:
            self.bias = jnp.zeros(in_features)

    def __call__(self, x):
        x = rearrange(x, '(nh dh) -> nh dh', nh=self.n_heads)
        x = einsum(x, self.weight, 'nh h, nh h dh -> nh dh')
        x = rearrange(x, 'nh dh -> (nh dh)')
        return x if self.bias is None else x + self.bias


@typecheck
def mlstm_forward(
    q: Float[Array, 'nh l dh'],
    k: Float[Array, 'nh l dh'], 
    v: Float[Array, 'nh l dh'],
    ig_pre_act: Float[Array, 'nh l'],
    fg_pre_act: Float[Array, 'nh l'],
    mask: Bool[Array, 'l l'],
) -> Float[Array, 'nh l dh']:

    _, l, dh = q.shape
     
    log_fg_act = jax.nn.log_sigmoid(fg_pre_act)  # (nh, l)
 
    log_fg_cumsum = jnp.pad(jnp.cumsum(log_fg_act, -1), ((0, 0), (1, 0)))  # (nh, l + 1)
     
    rep_log_fg_cumsum = jnp.repeat(log_fg_cumsum[..., None], l + 1, -1)  # (nh, l + 1) -> (nh, l + 1, l + 1)
 
    transpose_diff = rep_log_fg_cumsum - rep_log_fg_cumsum.transpose(0, 2, 1)  # (nh, l + 1, l + 1)
 
    log_fg = jnp.where(mask, transpose_diff[:, 1:, 1:], -jnp.inf)  # (nh, l, l)

    log_D = log_fg + ig_pre_act[:, None]

    max_log_D = jnp.max(log_D, -1, keepdims=True)
 
    log_D_stabilized = log_D - max_log_D  # (nh, l, l)
    D = jnp.exp(log_D_stabilized)
 
    scores = q @ (k / math.sqrt(dh)).transpose(0, 2, 1)  # (nh, l, l)

    c = scores * D

    norm = jnp.maximum(jnp.abs(jnp.sum(c, -1, keepdims=True)), jnp.exp(-max_log_D))

    c_normed = c / (norm + 1e-6)

    return c_normed @ v


@typecheck
def mlstm_step(
    q: Float[Array, 'nh dh'],
    k: Float[Array, 'nh dh'], 
    v: Float[Array, 'nh dh'],
    ig_pre_act: Float[Array, 'nh'],
    fg_pre_act: Float[Array, 'nh'],
    c_state: Float[Array, 'nh dh dh'],
    n_state: Float[Array, 'nh dh'],
    m_state: Float[Array, 'nh']
) -> Tuple[
   Float[Array, 'nh dh'],
   Tuple[Float[Array, 'nh dh dh'], Float[Array, 'nh dh'], Float[Array, 'nh']]
]:
    nh, dh = q.shape

    log_fg_act = jax.nn.log_sigmoid(fg_pre_act)

    next_m_state = jnp.maximum(log_fg_act + m_state, ig_pre_act)

    fg_act = jnp.exp(log_fg_act + m_state - next_m_state)
    ig_act = jnp.exp(ig_pre_act - next_m_state)

    k = k / math.sqrt(dh)

    next_c_state = einsum(fg_act, c_state, 'nh, nh ... -> nh ...') + einsum(ig_act, k, v, 'nh, nh i, nh j -> nh i j')
    next_n_state = einsum(fg_act, n_state, 'nh, nh dh -> nh dh') + einsum(ig_act, k, 'nh, nh dh -> nh dh')

    qn = einsum(q, next_n_state, 'nh dh, nh dh -> nh')[:, None]
    mv = jnp.exp(-next_m_state)[:, None]

    h = einsum(q, next_c_state, 'nh i, nh i dh -> nh dh') / (jnp.maximum(jnp.abs(qn), mv) + 1e-6)

    return h, (next_c_state, next_n_state, next_m_state)


class xLSTMLayer(eqx.Module):
    xlstm_norm: nn.LayerNorm
    win: nn.Linear
    wout: nn.Linear
    conv_kernel: Array
    conv_bias: Array | None = None
    wq: HeadLinear
    wk: HeadLinear
    wv: HeadLinear
    igate: nn.Linear
    fgate: nn.Linear
    group_norm: nn.GroupNorm 
    skip: Array
    ffn_norm: nn.LayerNorm | None = None
    ffn: GatedFFN | None = None
    args: xLSTMArgs

    def __init__(self, key, args):
        self.args = args

        win_key, wout_key, conv_key, wq_key, wk_key, wv_key, igate_key, fgate_key, ffn_key = jax.random.split(key, 9)

        self.xlstm_norm = nn.LayerNorm(args.d_model, use_weight=True, use_bias=args.use_ln_bias)
        
        self.win = set_linear_weight(win_key, args.d_model, 2 * args.d_inner, args.use_proj_bias, 'small')
        if args.use_proj_bias:
            self.win = eqx.tree_at(pick_bias, self.win, jnp.zeros_like(self.win.bias))
        self.wout = set_linear_weight(wout_key, args.d_inner, args.d_model, False, 'wang', args.n_layers)

        self.conv_kernel = small_init(conv_key, (args.d_inner, args.d_conv), args.d_inner)
        if self.args.use_conv_bias:
            self.conv_bias = jnp.zeros(args.d_inner)

        self.wq = HeadLinear(wq_key, args.d_inner, args.n_heads, args.use_qkv_bias)
        self.wk = HeadLinear(wk_key, args.d_inner, args.n_heads, args.use_qkv_bias)
        self.wv = HeadLinear(wv_key, args.d_inner, args.n_heads, args.use_qkv_bias)
        
        self.igate = set_linear_weight(igate_key, args.d_inner * 3, args.n_heads, True, 'small')
        self.igate = eqx.tree_at(pick_bias, self.igate, jnp.linspace(0, 0.1, args.n_heads))
        self.fgate = set_linear_weight(fgate_key, args.d_inner * 3, args.n_heads, True, 'zero')
        self.fgate = eqx.tree_at(pick_bias, self.fgate, jnp.linspace(3, 6, args.n_heads))

        self.group_norm = nn.GroupNorm(args.n_heads, args.d_inner)
        self.skip = jnp.ones(args.d_inner)

        if args.use_ffn:
            self.ffn_norm = nn.LayerNorm(args.d_model, use_weight=True, use_bias=args.use_ln_bias)
            self.ffn = GatedFFN(
                ffn_key, args.d_model, args.d_hidden, args.use_ffn_bias, args.n_layers, args.activation
            )

    def __call__(self, x, mask):
        h = vmap(self.xlstm_norm)(x)
        x = x + self.xlstm(h, mask)
        if self.ffn is not None:
            h = vmap(self.ffn_norm)(x)
            x = x + vmap(self.ffn)(h)
        return x
    
    def xlstm(self, x, mask):
        # (seq_len, d_model) -> (seq_len, d_inner), (seq_len, d_inner)
        x, res = jnp.split(vmap(self.win)(x), [self.args.d_inner], -1)
        
        conv_input = jnp.pad(x, ((self.args.d_conv - 1, 0), (0, 0))).T
        xc = vmap(jnp.convolve, (0, 0, None))(conv_input, self.conv_kernel, 'valid').T 

        if self.conv_bias is not None:
            xc = xc + self.conv_bias

        xc = jax.nn.silu(xc)  # (seq_len, d_inner)
        
        # all (seq_len, d_inner) 
        q, k, v = vmap(self.wq)(xc), vmap(self.wk)(xc), vmap(self.wv)(x)
        gate_input = jnp.concatenate([q, k, v], -1)

        q = rearrange(q, 'l (nh dh) -> nh l dh', nh=self.args.n_heads)
        k = rearrange(k, 'l (nh dh) -> nh l dh', nh=self.args.n_heads)       
        v = rearrange(v, 'l (nh dh) -> nh l dh', nh=self.args.n_heads)
        
        # both (seq_len, 3 * d_inner) -> (seq_len, n_heads) -> (n_heads, seq_len)
        ig_pre_act = vmap(self.igate)(gate_input).T
        fg_pre_act = vmap(self.fgate)(gate_input).T

        h = mlstm_forward(q, k, v, ig_pre_act, fg_pre_act, mask)
        h = vmap(self.group_norm)(rearrange(h, 'nh l dh -> l (nh dh)')) + self.skip * xc
        
        # (seq_len, d_inner) -> (seq_len, d_model)
        h = vmap(self.wout)(h * jax.nn.silu(res))

        return h
    
    def step(self, x, state):
        h, next_state = self.xlstm_step(self.xlstm_norm(x), state)
        x = x + h
        if self.ffn is not None:
            h = self.ffn(self.ffn_norm(x))
            x = x + h
        return x, next_state

    def xlstm_step(self, x, state):
        if state is None:
            state = (
                jnp.zeros((self.args.d_inner, self.args.d_conv - 1)),
                (
                    jnp.zeros((self.args.n_heads, self.args.d_head, self.args.d_head)),
                    jnp.zeros((self.args.n_heads, self.args.d_head)),
                    jnp.zeros(self.args.n_heads)
                )
            )

        conv_state, cnm_state = state

        x, res = jnp.split(self.win(x), [self.args.d_inner], -1)

        conv_input = jnp.concatenate([conv_state, x[:, None]], -1)
        xc = jnp.vecdot(conv_input, jnp.flip(self.conv_kernel, -1))

        if self.conv_bias is not None:
            xc = xc + self.conv_bias

        xc = jax.nn.silu(xc)

        q, k, v = self.wq(xc), self.wk(xc), self.wv(x)
        gate_input = jnp.concatenate([q, k, v], -1)

        q = rearrange(q, '(nh dh) -> nh dh', nh=self.args.n_heads)
        k = rearrange(k, '(nh dh) -> nh dh', nh=self.args.n_heads)
        v = rearrange(v, '(nh dh) -> nh dh', nh=self.args.n_heads)

        ig_pre_act = self.igate(gate_input)
        fg_pre_act = self.fgate(gate_input)

        h, next_cnm_state = mlstm_step(q, k, v, ig_pre_act, fg_pre_act, *cnm_state)
        h = self.group_norm(rearrange(h, 'nh dh -> (nh dh)')) + self.skip * xc

        h = self.wout(h * jax.nn.silu(res))
        next_state = (conv_input[:, 1:], next_cnm_state)

        return h, next_state


class xLSTM(eqx.Module):
    wte: Array
    layers: List[xLSTMLayer]
    norm: nn.LayerNorm
    wout: Array | None = None
    args: xLSTMArgs

    def __init__(self, key, args):
        wte_key, layers_key, wout_key = jax.random.split(key, 3)
        self.args = args
        self.wte = small_init(wte_key, (args.vocab_size, args.d_model), args.d_model)
        self.layers = [xLSTMLayer(subkey, args) for subkey in jax.random.split(layers_key, args.n_layers)]
        self.norm = nn.LayerNorm(args.d_model, use_weight=True, use_bias=True)
        
        if not args.tie_weights:
            self.wout = small_init(wout_key, (args.d_model, args.vocab_size), args.d_model)

    def __call__(self, tokens):
        x = self.wte[tokens]
        mask = jnp.bool(jnp.tril(jnp.ones((self.args.seq_len, self.args.seq_len))))

        for layer in self.layers:
            x = layer(x, mask)

        x = vmap(self.norm)(x)
        
        if self.wout is None:
            logits = x @ self.wte.T
        else: 
            logits = x @ self.wout

        return logits
    
    @filter_jit
    def step(self, token, state):
        if state is None:
            state = [None] * self.args.n_layers

        x = self.wte[token]

        next_state = []
        for layer, layer_state in zip(self.layers, state):
            x, next_layer_state = layer.step(x, layer_state)
            next_state.append(next_layer_state)

        x = self.norm(x)

        if self.wout is None:
            logits = x @ self.wte.T
        else:
            logits = x @ self.wout

        return logits, next_state


def test():
    key = jax.random.key(0)

    args = xLSTMArgs(
        d_model=32,
        n_heads=4,
        n_layers=4,
        vocab_size=128,
        seq_len=512,
        activation=jax.nn.gelu,
        use_ffn=True,
        tie_weights=False
    )

    model_key, tokens_key = jax.random.split(key)

    batch_size = 4

    model = xLSTM(model_key, args)

    tokens = jax.random.randint(tokens_key, (batch_size, args.seq_len), 0, args.vocab_size) 
    
    # compute logits in parallel 
    logits_batch = vmap(model)(tokens)
    
    # compute logits sequentially 
    state = None
    for token in tokens.T:
        logits, state = vmap(model.step)(token, state) 
     
    assert jnp.allclose(logits_batch[:, -1], logits, rtol=1e-3, atol=1e-5)
    
    print(logits_batch[:, -1])
    print()
    print(logits)
    print()
    print('parallel and sequential passes are consistent')


if __name__ == '__main__':
    test()

