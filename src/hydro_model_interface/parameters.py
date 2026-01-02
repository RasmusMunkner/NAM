from jax import numpy as jnp

def inv_sigmoid(x):
    eps = 1e-6
    x = jnp.clip(x, eps, 1 - eps)
    return jnp.log(x) - jnp.log1p(-x)

def inv_softplus(x):
    eps = 1e-6
    x = jnp.maximum(x, eps)
    return x + jnp.log(-jnp.expm1(-x))