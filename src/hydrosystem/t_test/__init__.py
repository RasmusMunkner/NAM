from jax import numpy as jnp
import scipy
from jaxtyping import Float
from typing import Any

def t_test(x0: Float[Any, "dim"], x1: Float[Any, "dim"]) -> Float[Any, ""]:
    """Perform a t-test for identical group means.

    Notes:
        The null-hypothesis is H0: x0 and x1 are i.i.d. normals with identical means, but possibly different variances.
    """
    score = (jnp.mean(x0) - jnp.mean(x1)) / jnp.sqrt(jnp.std(x0) / x0.shape[0] + jnp.std(x1) / x1.shape[0])
    df = x0.shape[0] + x1.shape[0] - 2
    p = scipy.stats.t.cdf(-jnp.abs(score), df) + (1-scipy.stats.t.cdf(jnp.abs(score), df))
    return p
