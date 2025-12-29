import jax
from jax import numpy as jnp
from typing import NamedTuple

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
    
def pack_params(**kwargs) -> dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v) for k, v in kwargs.items()}

class NAM_Observation(NamedTuple):
    """Class for holding input data to NAM."""
    p: jnp.ndarray # Precipitation (mm/d)
    epot: jnp.ndarray # Potential Evapotranspiration (mm/d)
    t: jnp.ndarray # Temperature (C)

class NAM_Parameters(NamedTuple):
    """Class for holding NAM parameters."""
    area: jnp.ndarray # Area of catchment
    
    c_area: jnp.ndarray # Catchment water gain/loss ratio
    c_snow: jnp.ndarray # Snow ratio
    
    l_max: jnp.ndarray # Lower zone water storage capacity
    u_max: jnp.ndarray # Surface water storage capacity
    
    cqof: jnp.ndarray # Overland flow coefficient
    ckif: jnp.ndarray # Interflow coefficient
    
    tof: jnp.ndarray # Overland flow threshold
    tif: jnp.ndarray # Interflow threshold
    tg: jnp.ndarray # Groundwater flow threshold
    
    ck1: jnp.ndarray # Flow timing coefficient 1
    ck2: jnp.ndarray # Flow timing coefficient 2
    ckbf: jnp.ndarray # Baseflow timing coefficient
    
class NAM_State(NamedTuple):
    """Class for holding NAM starting conditions."""
    s: jnp.ndarray # Snow storage
    u_ratio: jnp.ndarray # Surface storage, u = u_max * u_ratio
    l_ratio: jnp.ndarray # Lower zone storage, l = l_max * l_ratio
    
    qr1: jnp.ndarray
    qr2: jnp.ndarray
    bf: jnp.ndarray
