import jax
from jax import numpy as jnp

UNIT = {"cqof", "ckif", "tof", "tif", "tg", "ck1", "ck2", "ckbf", "c_snow", "u_ratio", "l_ratio"}
POSITIVE = {"area", "c_area", "u_max", "l_max", "s", "qr1", "qr2", "bf"}


def to_physical(d: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    return {
        k: (
            jax.nn.sigmoid(v) if k in UNIT
            else jax.nn.softplus(v) if k in POSITIVE
            else v
        )
        for k, v in d.items()
    }


def inv_sigmoid(x):
    eps = 1e-6
    x = jnp.clip(x, eps, 1 - eps)
    return jnp.log(x) - jnp.log1p(-x)


def inv_softplus(x):
    eps = 1e-6
    x = jnp.maximum(x, eps)
    return x + jnp.log(-jnp.expm1(-x))


def to_unconstrained(d: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    return {
        k: (
            inv_sigmoid(v) if k in UNIT
            else inv_softplus(v) if k in POSITIVE
            else v
        )
        for k, v in d.items()
    }