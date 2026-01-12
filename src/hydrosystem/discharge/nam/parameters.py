import jax
from jax import numpy as jnp
from jax import random
import distrax
from hydrosystem.discharge.nam.math import inv_sigmoid, inv_softplus, threshold_linear

from typing import NamedTuple


class NAMParameters(NamedTuple):
    area_: jnp.ndarray
    c_area_: jnp.ndarray
    cqof_: jnp.ndarray
    ckif_: jnp.ndarray
    tof_: jnp.ndarray
    tif_: jnp.ndarray
    tg_: jnp.ndarray
    ck1_: jnp.ndarray
    ck2_: jnp.ndarray
    ckbf_: jnp.ndarray
    c_snow_: jnp.ndarray
    u_max_: jnp.ndarray
    l_max_: jnp.ndarray

    s_: jnp.ndarray
    u_ratio_: jnp.ndarray
    l_ratio_: jnp.ndarray
    qr1_: jnp.ndarray
    qr2_: jnp.ndarray
    bf_: jnp.ndarray

    @classmethod
    def from_physical(
            cls,
            area: jnp.ndarray = 1055,
            c_area: jnp.ndarray = 0.9,
            cqof: jnp.ndarray = 0.3,
            ckif: jnp.ndarray = 1/20,
            tof: jnp.ndarray = 0.2,
            tif: jnp.ndarray = 0.5,
            tg: jnp.ndarray = 0.5,
            ck1: jnp.ndarray = jnp.exp(-1/2),
            ck2: jnp.ndarray = 0,
            ckbf: jnp.ndarray = jnp.exp(-1/500),
            c_snow: jnp.ndarray = 2,
            u_max: jnp.ndarray = 5,
            l_max: jnp.ndarray = 100,
            s: jnp.ndarray = 0,
            u_ratio: jnp.ndarray = 1,
            l_ratio: jnp.ndarray = 1,
            qr1: jnp.ndarray = 0.43,
            qr2: jnp.ndarray = 0,
            bf: jnp.ndarray = 0.86,
    ):
        return cls(
            area_=inv_softplus(area),
            c_area_=inv_softplus(c_area),
            cqof_=inv_sigmoid(cqof),
            ckif_=inv_sigmoid(ckif),
            tof_=inv_sigmoid(tof),
            tif_=inv_sigmoid(tif),
            tg_=inv_sigmoid(tg),
            ck1_=inv_sigmoid(ck1),
            ck2_=inv_sigmoid(ck2),
            ckbf_=inv_sigmoid(ckbf),
            c_snow_=inv_softplus(c_snow),
            u_max_=inv_softplus(u_max),
            l_max_=inv_softplus(l_max),
            s_=inv_softplus(s),
            u_ratio_=inv_sigmoid(u_ratio),
            l_ratio_=inv_sigmoid(l_ratio),
            qr1_=inv_softplus(qr1),
            qr2_=inv_softplus(qr2),
            bf_=inv_softplus(bf)
        )

    @classmethod
    def sample(
            cls,
            key: jax.Array = None,
            shape: tuple[int] = (1,),
            area: distrax.Distribution = distrax.Deterministic(1055),
            c_area: distrax.Distribution = distrax.Deterministic(0.9),
            cqof: distrax.Distribution = distrax.Uniform(0, 1),
            ckif: distrax.Distribution = distrax.Uniform(1 / 40, 1 / 20),
            tof: distrax.Distribution = distrax.Uniform(0, 0.9),
            tif: distrax.Distribution = distrax.Uniform(0, 0.9),
            tg: distrax.Distribution = distrax.Uniform(0, 0.9),
            ck1: distrax.Distribution = distrax.Uniform(0, 0.9),
            ck2: distrax.Distribution = distrax.Deterministic(0),
            ckbf: distrax.Distribution = distrax.Uniform(0.98, 0.993),
            c_snow: distrax.Distribution = distrax.Deterministic(2),
            u_max: distrax.Distribution = distrax.Uniform(5, 35),
            l_max: distrax.Distribution = distrax.Uniform(50, 500),
            s: distrax.Distribution = distrax.Deterministic(0),
            u_ratio: distrax.Distribution = distrax.Uniform(0, 1),
            l_ratio: distrax.Distribution = distrax.Uniform(0, 1),
            qr1: distrax.Distribution = distrax.Uniform(0, 1),
            qr2: distrax.Distribution = distrax.Deterministic(0),
            bf: distrax.Distribution = distrax.Uniform(0.5, 1)
    ):
        """Generate a NAM model.

        Parameters:
        -------------
        key: jax.Array
            Random key for the generation.
        shape : tuple[int]
            Shape of generated parameter arrays.

        Notes:
        -------------
            The trainable/fixed parameter split is set by default, but can be changed post-creation.
            To do this, use the freeze/unfreeze methods.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = random.split(key, 19)
        params = {
            "cqof": cqof.sample(seed=keys[0], sample_shape=shape),
            "ckif": ckif.sample(seed=keys[1], sample_shape=shape),
            "tof": tof.sample(seed=keys[2], sample_shape=shape),
            "tif": tif.sample(seed=keys[3], sample_shape=shape),
            "tg": tg.sample(seed=keys[4], sample_shape=shape),
            "ck1": ck1.sample(seed=keys[5], sample_shape=shape),
            "ck2": ck2.sample(seed=keys[6], sample_shape=shape),
            "ckbf": ckbf.sample(seed=keys[7], sample_shape=shape),
            "u_max": u_max.sample(seed=keys[8], sample_shape=shape),
            "l_max": l_max.sample(seed=keys[9], sample_shape=shape),
            "area": area.sample(seed=keys[10], sample_shape=shape),
            "c_area": c_area.sample(seed=keys[11], sample_shape=shape),
            "c_snow": c_snow.sample(seed=keys[12], sample_shape=shape),
            "s": s.sample(seed=keys[13], sample_shape=shape),
            "qr1": qr1.sample(seed=keys[14], sample_shape=shape),
            "qr2": qr2.sample(seed=keys[15], sample_shape=shape),
            "bf": bf.sample(seed=keys[16], sample_shape=shape),
            "u_ratio": u_ratio.sample(seed=keys[17], sample_shape=shape),
            "l_ratio": l_ratio.sample(seed=keys[18], sample_shape=shape)
        }
        return cls.from_physical(**params)


    @classmethod
    def default_freezemask(cls):
        return cls(
                area_=True,
                c_area_=True,
                cqof_=False,
                ckif_=False,
                tof_=False,
                tif_=False,
                tg_=False,
                ck1_=False,
                ck2_=True,
                ckbf_=False,
                c_snow_=True,
                u_max_=False,
                l_max_=False,
                s_=True,
                u_ratio_=False,
                l_ratio_=False,
                qr1_=False,
                qr2_=True,
                bf_=False
            )


    def update(self, updates: dict[str, jnp.ndarray]):
        can_be_updated_sigmoid = {"u_ratio", "l_ratio"}
        can_be_updated_softplus = {"s", "qr1", "qr2", "bf"}
        self_as_dict = self._asdict()
        for k in can_be_updated_sigmoid:
            if k in updates:
                self_as_dict[k + "_"] = inv_sigmoid(updates[k])
        for k in can_be_updated_softplus:
            if k in updates:
                self_as_dict[k + "_"] = inv_softplus(updates[k])

        return self.__class__(**self_as_dict)


    def total_water_stored(self):
        """Calculate the total amount of water released over an infinite horizon with no water input.

        The key part of the calculation is the routing parameters.
        Water stored in qr1/qr2/bf is released over an infinite horizon weighted by a geometric series.
        These stores are recharged via (interflow+overflow)*(1-w)*(geometric series), hence recharge passes through.
        This means that we can just route s/u/l right through with no weighting. However, the (1-w) factor is
        never applied to initial storage, which is especially important for baseflow. Hence adjustments are needed.
        The correct adjustment is bf0 * w/(1-w), since it is never multiplied by (1-w).
        """
        total = 0
        total += self.u_max * self.u_ratio
        total += self.l_max * self.l_ratio
        total += self.s
        total += self.ck1 / (1 - self.ck1) * self.qr1
        total += self.ck2 / (1 - self.ck2) * self.qr2
        total += self.ckbf / (1 - self.ckbf) * self.bf
        return total


    @property
    def area(self) -> jnp.ndarray:
        return jax.nn.softplus(self.area_)

    @property
    def c_area(self) -> jnp.ndarray:
        return jax.nn.softplus(self.c_area_)

    @property
    def cqof(self) -> jnp.ndarray:
        return jax.nn.sigmoid(self.cqof_)

    @property
    def ckif(self) -> jnp.ndarray:
        return jax.nn.sigmoid(self.ckif_)

    @property
    def tof(self) -> jnp.ndarray:
        return jax.nn.sigmoid(self.tof_)

    @property
    def tif(self) -> jnp.ndarray:
        return jax.nn.sigmoid(self.tif_)

    @property
    def tg(self) -> jnp.ndarray:
        return jax.nn.sigmoid(self.tg_)

    @property
    def ck1(self) -> jnp.ndarray:
        return jax.nn.sigmoid(self.ck1_)

    @property
    def ck2(self) -> jnp.ndarray:
        return jax.nn.sigmoid(self.ck2_)

    @property
    def ckbf(self) -> jnp.ndarray:
        return jax.nn.sigmoid(self.ckbf_)

    @property
    def c_snow(self) -> jnp.ndarray:
        return jax.nn.softplus(self.c_snow_)

    @property
    def u_max(self) -> jnp.ndarray:
        return jax.nn.softplus(self.u_max_)

    @property
    def l_max(self) -> jnp.ndarray:
        return jax.nn.softplus(self.l_max_)

    @property
    def s(self) -> jnp.ndarray:
        return jax.nn.softplus(self.s_)

    @property
    def u_ratio(self) -> jnp.ndarray:
        return jax.nn.sigmoid(self.u_ratio_)

    @property
    def l_ratio(self) -> jnp.ndarray:
        return jax.nn.sigmoid(self.l_ratio_)

    @property
    def qr1(self) -> jnp.ndarray:
        return jax.nn.softplus(self.qr1_)

    @property
    def qr2(self) -> jnp.ndarray:
        return jax.nn.softplus(self.qr2_)

    @property
    def bf(self) -> jnp.ndarray:
        return jax.nn.softplus(self.bf_)