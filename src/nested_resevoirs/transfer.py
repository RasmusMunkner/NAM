import jax
from jax import numpy as jnp
from jaxtyping import Float, Array

def scalar_transfer_function(
        x: Float[Array, ""],
        vx: Float[Array, "dim1"],
        vy: Float[Array, "dim1"],
        s: Float[Array, ""]
) -> Float[Array, ""]:
    """Linear, monotone interpolation on the unit interval.

    Parameters
    --------------
    x : jnp.ndarray
        Function argument. Should be a number in the unit interval.
    vx: jnp.ndarray
        Breakpoint weights. Breakpoints between linear segments are determined based
        on the cumulative sum of the softmax of vx.
    vy: jnp.ndarray
        Value weights. The value of the i'th breakpoint is determined as
        the i'th entry in the cumulated softmax of vy.
    s: jnp.ndarray
        Scale. The final result is scaled by the sigmoid of this number.

    Notes
    --------------
    The interpretation of this function is that it maps some portion of a hydrological resevoir
    as flow based on the value of one auxiliary variable.
    E.g. the amount of surface percolation is a proportion of the total available surface water determined
    by the saturation differential between surface and subsurface.

    The shape of the mapping is determined by vx, vy and s and the mapping is a.e. differentiable with respect to each
    of these parameters, thus permitting gradient based optimization.

    In practice, the levels seem to collapse often, which could lead to subpar performance and dying gradient.
    """
    xp = jnp.cumulative_sum(jax.nn.softmax(vx), include_initial=True)
    fp = jnp.cumulative_sum(jax.nn.softmax(vy), include_initial=True)
    linear = jnp.interp(x, xp, fp)
    return linear * jax.nn.sigmoid(s)

pmap_transfer_function = jax.jit(jax.vmap(scalar_transfer_function, in_axes=[0,0,0,0]))
parmap_transfer_function = jax.jit(jax.vmap(scalar_transfer_function, in_axes=[None,0,0,0]))
cross_transfer_function = jax.jit(
    jax.vmap(jax.vmap(scalar_transfer_function, in_axes=[None,0,0,0]), in_axes=[0,None,None,None])
)















