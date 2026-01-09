from jax import numpy as jnp
from jaxtyping import Float, Int
from typing import NamedTuple, Any

def double_mass_analysis(
        x0: Float[Any, "dim0"],
        x1: Float[Any, "dim1"],
        y0: Float[Any, "dim0"],
        y1: Float[Any, "dim1"]
) -> Float[Any, ""]:
    """Calculate the quotient of slopes between cumsum(x) and cumsum(y)."""
    def rm_nan(z):
        z = jnp.asarray(z)
        idx_nan = jnp.isnan(z)
        return z[~idx_nan]

    x0, y0, x1, y1 = rm_nan(x0), rm_nan(y0), rm_nan(x1), rm_nan(y1)
    x0, x1  = jnp.cumsum(x0).reshape(-1,1), jnp.cumsum(x1).reshape(-1,1)
    y0, y1 =  jnp.cumsum(y0).reshape(-1,1), jnp.cumsum(y1).reshape(-1,1)
    coef0 = jnp.linalg.inv(x0.T @ x0) @ x0.T @ y0
    coef1 = jnp.linalg.inv(x1.T @ x1) @ x1.T @ y1
    return coef1[0] / coef0[0]


def changepoint(x: Float[Any, "dim0"]) -> Int[Any, ""]:
    """Optimal L2 split. NaN's are ignored. Returns the index of the last item on the left side."""
    x = jnp.asarray(x)
    mask = ~jnp.isnan(x)
    x_no_nans = x[mask]
    n = x_no_nans.shape[0]

    # Need at least two non-NaN points to split
    if n < 2:
        return jnp.array(-1, dtype=jnp.int32)

    # Cumulative sums s_k for k=1..n
    s = jnp.cumsum(x_no_nans)         # shape (n,)
    S = s[-1]

    # Evaluate splits at k = 1..n-1 so both sides are non-empty
    k = jnp.arange(1, n)              # counts on the left
    s_left = s[:-1]                   # s_k for k = 1..n-1
    # L2 gain (up to constants): s_left^2 / k + (S - s_left)^2 / (n - k)
    gain = (s_left ** 2) / k + ((S - s_left) ** 2) / (n - k)

    # Best split index among compacted array
    best_split_compact = jnp.argmax(gain)

    # Map back to original indices (before NaN removal)
    original_idxs = jnp.nonzero(mask)[0]   # length n
    return original_idxs[best_split_compact + 1]


def double_mass_analysis_with_changepoint(
        x: Float[Any, "dim0"],
        y: Float[Any, "dim0"],
) -> Float[Any, "dim0"]:
    """Correct initial part of x such that y/x is approximately constant."""
    x, y = jnp.asarray(x), jnp.asarray(y)
    best_split = changepoint(y/x)
    correction = double_mass_analysis(x[best_split:], x[:best_split], y[best_split:], y[:best_split])
    return jnp.concatenate([jnp.ones_like(x[:best_split])*correction, jnp.ones_like(x[best_split:])])
