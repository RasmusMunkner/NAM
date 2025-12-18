from jax import numpy as jnp

def condition(saturation: jnp.ndarray, threshold: jnp.ndarray) -> jnp.ndarray:
    return (saturation - threshold) / (1 - threshold) * (saturation > threshold)