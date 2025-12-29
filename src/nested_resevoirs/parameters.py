import jax
from jax import numpy as jnp
from jax import random
from typing import NamedTuple

from matplotlib import pyplot as plt


class NRState(NamedTuple):
    snow: jnp.ndarray
    surface: jnp.ndarray
    subsurface: jnp.ndarray
    groundwater: jnp.ndarray


class TransferFunction(NamedTuple):
    vx: jnp.ndarray # of shape (k,)
    vy: jnp.ndarray # of shape (k,)
    s: jnp.ndarray # of shape (1,)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the transfer function.

        Parameters:
            x (jnp.ndarray): the input variable. Must be between 0 and 1.

        Notes:
            The transfer function is defined as the linear interpolation of the cumulative sums of
            the softmax of (vx,vy) scaled by sigmoid(s).
        """
        qx = jnp.cumulative_sum(jax.nn.softmax(self.vx), include_initial=True)
        qy = jnp.cumulative_sum(jax.nn.softmax(self.vy), include_initial=True)
        idx = jnp.minimum(jnp.searchsorted(qx, x, side="right"), qx.shape[0]-1)
        w = (x - qx[idx-1]) / (qx[idx]-qx[idx-1])
        linear = qy[idx-1] * (1 - w) + qy[idx] * w
        return linear * jax.nn.sigmoid(self.s)


    @classmethod
    def sample(cls, key, df: int = 3, scale: tuple[float, float] = (0,1)):
        vkey, scalekey = random.split(key, 2)
        v = jax.random.normal(vkey, shape=(2*(df+1),))
        scale = inv_sigmoid(random.uniform(scalekey, minval=scale[0], maxval=scale[1]))
        return cls(v[:(df+1)], v[(df+1):], scale)


    def show(self, ax=None):
        xx = jnp.linspace(0,1,100)
        yy = self(xx)
        if ax is None:
            plt.plot(xx, yy)
        else:
            ax.plot(xx, yy)


class NRParameters(NamedTuple):

    surface_storage_: jnp.ndarray
    subsurface_storage_: jnp.ndarray
    groundwater_storage_: jnp.ndarray

    snowmelt: TransferFunction
    surface_flow: TransferFunction
    surface_percolation: TransferFunction
    subsurface_flow: TransferFunction
    subsurface_percolation: TransferFunction
    groundwater_flow: TransferFunction

    @property
    def surface_storage(self) -> jnp.ndarray:
        return jax.nn.softplus(self.surface_storage_)

    @property
    def subsurface_storage(self) -> jnp.ndarray:
        return jax.nn.softplus(self.subsurface_storage_)

    @property
    def groundwater_storage(self) -> jnp.ndarray:
        return jax.nn.softplus(self.groundwater_storage_)


    @classmethod
    def sample(
            cls,
            key,
            surface_storage: tuple[float, float] = (5,50),
            subsurface_storage: tuple[float, float] = (100,500),
            groundwater_storage: tuple[float, float] = (5000, 500000),
            snowmelt_df: int = 1,
            snowmelt_scale: tuple[float, float] = (0.5, 1),
            surface_flow_df: int = 3,
            surface_flow_scale: tuple[float, float] = (0, 0.9),
            surface_percolation_df: int = 3,
            surface_percolation_scale: tuple[float, float] = (0, 0.5),
            subsurface_flow_df: int = 3,
            subsurface_flow_scale: tuple[float, float] = (0,0.2),
            subsurface_percolation_df: int = 3,
            subsurface_percolation_scale: tuple[float, float] = (0, 0.2),
            groundwater_flow_df: int = 3,
            groundwater_flow_scale: tuple[float, float] = (0, 0.01),
    ):
        keys = random.split(key, 9)
        return cls(
            surface_storage_= inv_softplus(
                random.uniform(keys[0], minval=surface_storage[0], maxval=surface_storage[1])
            ),
            subsurface_storage_= inv_softplus(
                random.uniform(keys[1], minval=subsurface_storage[0], maxval=subsurface_storage[1])
            ),
            groundwater_storage_= inv_softplus(
                random.uniform(keys[2], minval=groundwater_storage[0], maxval=groundwater_storage[1])
            ),
            snowmelt=TransferFunction.sample(keys[3], snowmelt_df, snowmelt_scale),
            surface_flow=TransferFunction.sample(keys[4], surface_flow_df, surface_flow_scale),
            surface_percolation=TransferFunction.sample(keys[5], surface_percolation_df, surface_percolation_scale),
            subsurface_flow=TransferFunction.sample(keys[6], subsurface_flow_df, subsurface_flow_scale),
            subsurface_percolation=TransferFunction.sample(keys[7], subsurface_percolation_df, subsurface_percolation_scale),
            groundwater_flow=TransferFunction.sample(keys[8], groundwater_flow_df, groundwater_flow_scale)
        )


def inv_sigmoid(x):
    eps = 1e-6
    x = jnp.clip(x, eps, 1 - eps)
    return jnp.log(x) - jnp.log1p(-x)


def inv_softplus(x):
    eps = 1e-6
    x = jnp.maximum(x, eps)
    return x + jnp.log(-jnp.expm1(-x))

























