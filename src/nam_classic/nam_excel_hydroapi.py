import jax
from jax import numpy as jnp
from jax import random
import optax
import distrax
from hydro_model_interface import HydroModel, HydroObservation
from hydro_model_interface.parameters import inv_softplus, inv_sigmoid
from nam_classic.utils import condition
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
            area: jnp.ndarray,
            c_area: jnp.ndarray,
            cqof: jnp.ndarray,
            ckif: jnp.ndarray,
            tof: jnp.ndarray,
            tif: jnp.ndarray,
            tg: jnp.ndarray,
            ck1: jnp.ndarray,
            ck2: jnp.ndarray,
            ckbf: jnp.ndarray,
            c_snow: jnp.ndarray,
            u_max: jnp.ndarray,
            l_max: jnp.ndarray,
            s: jnp.ndarray,
            u_ratio: jnp.ndarray,
            l_ratio: jnp.ndarray,
            qr1: jnp.ndarray,
            qr2: jnp.ndarray,
            bf: jnp.ndarray,
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
            qr1_=inv_sigmoid(qr1),
            qr2_=inv_sigmoid(qr2),
            bf_=inv_sigmoid(bf)
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
        return jax.nn.sigmoid(self.qr1_)

    @property
    def qr2(self) -> jnp.ndarray:
        return jax.nn.sigmoid(self.qr2_)

    @property
    def bf(self) -> jnp.ndarray:
        return jax.nn.sigmoid(self.bf_)


class NAM(HydroModel):
    """Exact replication of the NAM model as it was implemented in excel."""

    def __init__(
            self,
            params: NAMParameters,
            frozen: NamedTuple = None,
            optimizer: optax.GradientTransformationExtraArgs = None,
    ):
        if frozen is None:
            frozen = NAMParameters(
                area_=True,
                c_area_=True,
                cqof_=False,
                ckif_=False,
                tof_=False,
                tif_=False,
                tg_=False,
                ck1_=False,
                ck2_=True,
                ckbf_=True,
                c_snow_=True,
                u_max_=False,
                l_max_=False,
                s_=True,
                u_ratio_=False,
                l_ratio_=False,
                qr1_=False,
                qr2_=True,
                bf_=True
            )
        super().__init__(params, frozen, optimizer)


    @staticmethod
    def step(
            params: NAMParameters,
            obs: HydroObservation
    ):
        # Decide if precipitation is rain or snow
        rain, snow = obs.p * (obs.t > 0), obs.p * (obs.t < 0)  # Note the mistake for t=0.

        # Settle snow budget
        snowmelt = jnp.maximum(0, jnp.minimum(params.c_snow * obs.t, params.s))
        s_out = params.s + snow - snowmelt

        # Settle surface water budget
        u = params.u_ratio * params.u_max
        u_budget = u + rain + snowmelt  # total water available after rain+snowmelt
        u_budget_no_rain = u + snowmelt  # total water available, discounting rain

        e_p = jnp.minimum(u_budget_no_rain, obs.epot)  # Follows excel, but I think it should include rain
        u_budget -= e_p
        # min of (estimate, energy constraint, (mass constraint???, missing???)) - Likely ok, talked to lecturer
        e_a = jnp.minimum(params.l_ratio * obs.epot,obs.epot - e_p)

        # Follows excel, but why not just use u_budget?
        q_interflow = params.ckif * condition(params.l_ratio,params.tif) * u
        q_interflow = jnp.minimum(q_interflow, u_budget_no_rain - e_p)
        u_budget -= q_interflow

        excess = jnp.maximum(0, u_budget - params.u_max)
        q_overflow = params.cqof * condition(params.l_ratio, params.tof) * excess
        u_ratio_out = (u_budget - excess) / params.u_max

        # Settle lower zone budget
        percolation = (excess - q_overflow) * condition(params.l_ratio, params.tg)
        dl = excess - q_overflow - percolation
        l_ratio_out = params.l_ratio + (dl - e_a) / params.l_max

        # Calculate flows
        qr1_out = params.qr1 * params.ck1 + (q_overflow + q_interflow) * (1 - params.ck1)
        qr2_out = params.qr2 * params.ck2 + qr1_out * (1 - params.ck2)
        bf_out = params.bf * params.ckbf + percolation * (1 - params.ckbf) * params.c_area

        # Calculate simulated discharge
        # params.area rescaled from mm/d to m3/s. Note the /86.4 handles the time rescaling.
        qsim = (qr2_out + bf_out) * params.area / 86.4

        # Return
        next_state = params._asdict()
        next_state.update({
            "s_": inv_softplus(s_out), "u_ratio_": inv_sigmoid(u_ratio_out), "l_ratio_": inv_sigmoid(l_ratio_out),
            "qr1_": inv_sigmoid(qr1_out), "qr2_": inv_sigmoid(qr2_out), "bf_": inv_sigmoid(bf_out)
        })
        return NAMParameters(**next_state), qsim















