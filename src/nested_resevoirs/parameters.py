import jax
from jax import numpy as jnp
from jax import random
from typing import NamedTuple
import distrax

from matplotlib import pyplot as plt

from hydro_model_interface.parameters import inv_softplus, inv_sigmoid
from nested_resevoirs.transfer import pmap_transfer_function, cross_transfer_function, parmap_transfer_function


class TransferFunctionParams(NamedTuple):
    vx: jnp.ndarray # of shape (k,)
    vy: jnp.ndarray # of shape (k,)
    s: jnp.ndarray # of shape (1,)

    def pmap(self, x: jnp.ndarray ) -> jnp.ndarray:
        return pmap_transfer_function(x, self.vx, self.vy, self.s)

    def cross(self, x: jnp.ndarray ) -> jnp.ndarray:
        return cross_transfer_function(x, self.vx, self.vy, self.s)

    def parmap(self, x: jnp.ndarray) -> jnp.ndarray:
        return parmap_transfer_function(x, self.vx, self.vy, self.s)

    @classmethod
    def sample(
            cls,
            key: jax.Array,
            shape: tuple[int] = (1,),
            df: distrax.Distribution = distrax.Deterministic(3),
            scale: distrax.Distribution = distrax.Uniform(0,1)
    ):
        dfkey, vkey, scalekey = random.split(key, 3)
        df_realized = int(df.sample(seed=dfkey)) # Needs to be a 0D integer
        v = jax.random.normal(vkey, shape=(*shape, 2*(df_realized+1)))
        scale_realized = inv_sigmoid(scale.sample(seed=scalekey, sample_shape=shape))
        return cls(v[...,:(df_realized+1)], v[...,(df_realized+1):], scale_realized)


    def show(self, ax=None):
        xx = jnp.linspace(0,1,100)
        yy = cross_transfer_function(xx, self.vx, self.vy, self.s)
        for i in range(yy.shape[-1]):
            if ax is None:
                plt.plot(xx, yy[:,i])
            else:
                ax.plot(xx, yy[:,i])


class NRParameters(NamedTuple):

    # State
    snow_amount_: jnp.ndarray
    surface_ratio_: jnp.ndarray
    subsurface_ratio_: jnp.ndarray
    groundwater_ratio_: jnp.ndarray

    # Parameters
    surface_storage_: jnp.ndarray
    subsurface_storage_: jnp.ndarray
    groundwater_storage_: jnp.ndarray

    snowmelt: TransferFunctionParams
    surface_flow: TransferFunctionParams
    surface_percolation: TransferFunctionParams
    subsurface_flow: TransferFunctionParams
    subsurface_percolation: TransferFunctionParams
    groundwater_flow: TransferFunctionParams

    @property
    def snow_amount(self) -> jnp.ndarray:
        return jax.nn.softplus(self.snow_amount_)

    @property
    def surface_amount(self):
        return jax.nn.sigmoid(self.surface_ratio_) * jax.nn.softplus(self.surface_storage_)

    @property
    def subsurface_amount(self):
        return jax.nn.sigmoid(self.subsurface_ratio_) * jax.nn.softplus(self.subsurface_storage_)

    @property
    def groundwater_amount(self):
        return jax.nn.sigmoid(self.groundwater_ratio_) * jax.nn.softplus(self.groundwater_storage_)

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
            key: jax.Array = None,
            shape: tuple[int] = (1,),
            surface_storage: distrax.Distribution = distrax.Uniform(5,50),
            subsurface_storage: distrax.Distribution = distrax.Uniform(100,500),
            groundwater_storage: distrax.Distribution = distrax.Uniform(5000, 500000),
            snowmelt_df: distrax.Distribution = distrax.Deterministic(1),
            snowmelt_scale: distrax.Distribution = distrax.Uniform(0.5, 1),
            surface_flow_df: distrax.Distribution = distrax.Deterministic(3),
            surface_flow_scale: distrax.Distribution = distrax.Uniform(0, 0.9),
            surface_percolation_df: distrax.Distribution = distrax.Deterministic(3),
            surface_percolation_scale: distrax.Distribution = distrax.Uniform(0, 0.5),
            subsurface_flow_df: distrax.Distribution = distrax.Deterministic(3),
            subsurface_flow_scale: distrax.Distribution = distrax.Uniform(0,0.2),
            subsurface_percolation_df: distrax.Distribution = distrax.Deterministic(3),
            subsurface_percolation_scale: distrax.Distribution = distrax.Uniform(0, 0.2),
            groundwater_flow_df: distrax.Distribution = distrax.Deterministic(3),
            groundwater_flow_scale: distrax.Distribution = distrax.Uniform(0, 1e-5),
            snow_amount: distrax.Distribution = distrax.Deterministic(0),
            surface_ratio: distrax.Distribution = distrax.Uniform(0,1),
            subsurface_ratio: distrax.Distribution = distrax.Uniform(0,1),
            groundwater_ratio: distrax.Distribution = distrax.Uniform(0,1),
    ):
        if key is None:
            key = random.PRNGKey(0)
        keys = random.split(key, 13)
        return cls(
            surface_storage_= inv_softplus(
                surface_storage.sample(seed=keys[0], sample_shape=shape),
            ),
            subsurface_storage_= inv_softplus(
                subsurface_storage.sample(seed=keys[1], sample_shape=shape),
            ),
            groundwater_storage_= inv_softplus(
                groundwater_storage.sample(seed=keys[2], sample_shape=shape),
            ),
            snowmelt=TransferFunctionParams.sample(
                keys[3], shape, snowmelt_df, snowmelt_scale
            ),
            surface_flow=TransferFunctionParams.sample(
                keys[4], shape, surface_flow_df, surface_flow_scale
            ),
            surface_percolation=TransferFunctionParams.sample(
                keys[5], shape, surface_percolation_df, surface_percolation_scale
            ),
            subsurface_flow=TransferFunctionParams.sample(
                keys[6], shape, subsurface_flow_df, subsurface_flow_scale
            ),
            subsurface_percolation=TransferFunctionParams.sample(
                keys[7], shape, subsurface_percolation_df, subsurface_percolation_scale
            ),
            groundwater_flow=TransferFunctionParams.sample(
                keys[8], shape, groundwater_flow_df, groundwater_flow_scale
            ),
            snow_amount_=inv_softplus(
                snow_amount.sample(seed=keys[9], sample_shape=shape),
            ),
            surface_ratio_=inv_sigmoid(
                surface_ratio.sample(seed=keys[10], sample_shape=shape)
            ),
            subsurface_ratio_=inv_sigmoid(
                subsurface_ratio.sample(seed=keys[11], sample_shape=shape)
            ),
            groundwater_ratio_=inv_sigmoid(
                groundwater_ratio.sample(seed=keys[12], sample_shape=shape)
            )
        )
























