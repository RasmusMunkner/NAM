from jax import numpy as jnp

def inv_sigmoid(x: jnp.ndarray):
    x = jnp.asarray(x, dtype=float)
    eps = jnp.finfo(x.dtype).resolution
    x = jnp.clip(x, eps, 1 - eps)
    return jnp.log(x) - jnp.log1p(-x)

def inv_softplus(x: jnp.ndarray):
    x = jnp.asarray(x, dtype=float)
    eps = jnp.finfo(x.dtype).resolution if hasattr(x, 'dtype') else 1e-6
    x = jnp.maximum(x, eps)
    return x + jnp.log(-jnp.expm1(-x))