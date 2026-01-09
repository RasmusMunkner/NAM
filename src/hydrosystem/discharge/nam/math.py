from jax import numpy as jnp

def threshold_linear(x: jnp.ndarray, threshold: jnp.ndarray) -> jnp.ndarray:
    return (x - threshold) / (1 - threshold) * (x > threshold)

def inv_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    eps = jnp.finfo(x.dtype).resolution
    x = jnp.clip(x, eps, 1 - eps)
    return jnp.log(x) - jnp.log1p(-x)

def inv_softplus(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    eps = jnp.finfo(x.dtype).resolution if hasattr(x, 'dtype') else 1e-6
    x = jnp.maximum(x, eps)
    return x + jnp.log(-jnp.expm1(-x))