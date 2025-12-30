import jax
from jax import numpy as jnp
from jax import random
import distrax
from hydro_model_interface import HydroModel, HydroObservation
from nam_classic.utils import condition

parameters = {"area", "c_area", "cqof", "ckif", "tof", "tif", "tg", "ck1", "ck2", "ckbf", "c_snow", "u_max", "l_max"}
state_variables = {"s", "qr1", "qr2", "bf", "u_ratio", "l_ratio"}

class NAM(HydroModel):
    """Exact replication of the NAM model as it was implemented in excel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for p in parameters:
            if p not in self.params:
                raise ValueError(f"Parameter {p} is not defined")
        for sv in state_variables:
            if sv not in self.state:
                raise ValueError(f"State variable {sv} is not defined")

    @classmethod
    def generate(
            cls,
            key: jax.Array = None,
            shape: tuple[int] = (1,),
            area: distrax.Distribution = distrax.Deterministic(1055),
            c_area: distrax.Distribution = distrax.Deterministic(0.9),
            cqof: distrax.Distribution = distrax.Uniform(0, 1),
            ckif: distrax.Distribution = distrax.Uniform(1/40, 1/20),
            tof: distrax.Distribution = distrax.Uniform(0, 0.9),
            tif: distrax.Distribution = distrax.Uniform(0, 0.9),
            tg: distrax.Distribution = distrax.Uniform(0, 0.9),
            ck1: distrax.Distribution = distrax.Uniform(0,0.9),
            ck2: distrax.Distribution = distrax.Deterministic(0),
            ckbf: distrax.Distribution = distrax.Uniform(0.98, 0.993),
            c_snow: distrax.Distribution = distrax.Deterministic(2),
            u_max: distrax.Distribution = distrax.Uniform(5, 35),
            l_max: distrax.Distribution = distrax.Uniform(50, 500),
            s: distrax.Distribution = distrax.Deterministic(0),
            qr1: distrax.Distribution = distrax.Uniform(0, 1),
            qr2: distrax.Distribution = distrax.Deterministic(0),
            bf: distrax.Distribution = distrax.Uniform(0.5,1),
            u_ratio: distrax.Distribution = distrax.Uniform(0,1),
            l_ratio: distrax.Distribution = distrax.Uniform(0,1)
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
        params_trainable = {
            "cqof": cqof.sample(seed=keys[0], sample_shape=shape),
            "ckif": ckif.sample(seed=keys[1], sample_shape=shape),
            "tof": tof.sample(seed=keys[2], sample_shape=shape),
            "tif": tif.sample(seed=keys[3], sample_shape=shape),
            "tg": tg.sample(seed=keys[4], sample_shape=shape),
            "ck1": ck1.sample(seed=keys[5], sample_shape=shape),
            "ck2": ck2.sample(seed=keys[6], sample_shape=shape),
            "ckbf": ckbf.sample(seed=keys[7], sample_shape=shape),
            "u_max": u_max.sample(seed=keys[8], sample_shape=shape),
            "l_max": l_max.sample(seed=keys[9], sample_shape=shape)
        }
        params_fixed = {
            "area": area.sample(seed=keys[10], sample_shape=shape),
            "c_area": c_area.sample(seed=keys[11], sample_shape=shape),
            "ck2": ck2.sample(seed=keys[12], sample_shape=shape),
            "c_snow": c_snow.sample(seed=keys[13], sample_shape=shape)
        }
        state_trainable = {
            "qr1": qr1.sample(seed=keys[15], sample_shape=shape),
            "bf": bf.sample(seed=keys[17], sample_shape=shape),
            "u_ratio": u_ratio.sample(seed=keys[18], sample_shape=shape),
            "l_ratio": l_ratio.sample(seed=keys[19], sample_shape=shape)
        }
        state_fixed = {
            "s": s.sample(seed=keys[14], sample_shape=shape),
            "qr2": qr2.sample(seed=keys[16], sample_shape=shape),
        }
        return cls(params_trainable, state_trainable, params_fixed, state_fixed)


    @staticmethod
    def step(
            params: dict[str, jnp.ndarray],
            state: dict[str, jnp.ndarray],
            obs: HydroObservation
    ):
        # Decide if precipitation is rain or snow
        rain, snow = obs.p * (obs.t > 0), obs.p * (obs.t < 0)  # Note the mistake for t=0.

        # Settle snow budget
        snowmelt = jnp.maximum(0, jnp.minimum(params["c_snow"] * obs.t, state["s"]))
        s_out = state["s"] + snow - snowmelt

        # Settle surface water budget
        u = state["u_ratio"] * params["u_max"]
        u_budget = u + rain + snowmelt  # total water available after rain+snowmelt
        u_budget_no_rain = u + snowmelt  # total water available, discounting rain

        e_p = jnp.minimum(u_budget_no_rain, obs.epot)  # Follows excel, but I think it should include rain
        u_budget -= e_p
        # min of (estimate, energy constraint, (mass constraint???, missing???)) - Likely ok, talked to lecturer
        e_a = jnp.minimum(state["l_ratio"] * obs.epot,obs.epot - e_p)

        # Follows excel, but why not just use u_budget?
        q_interflow = params["ckif"] * condition(state["l_ratio"],params["tif"]) * u
        q_interflow = jnp.minimum(q_interflow, u_budget_no_rain - e_p)
        u_budget -= q_interflow

        excess = jnp.maximum(0, u_budget - params["u_max"])
        q_overflow = params["cqof"] * condition(state["l_ratio"], params["tof"]) * excess
        u_ratio_out = (u_budget - excess) / params["u_max"]

        # Settle lower zone budget
        percolation = (excess - q_overflow) * condition(state["l_ratio"], params["tg"])
        dl = excess - q_overflow - percolation
        l_ratio_out = state["l_ratio"] + (dl - e_a) / params["l_max"]

        # Calculate flows
        qr1_out = state["qr1"] * params["ck1"] + (q_overflow + q_interflow) * (1 - params["ck1"])
        qr2_out = state["qr2"] * params["ck2"] + qr1_out * (1 - params["ck2"])
        bf_out = state["bf"] * params["ckbf"] + percolation * (1 - params["ckbf"]) * params["c_area"]

        # Calculate simulated discharge
        # params.area rescaled from mm/d to m3/s. Note the /86.4 handles the time rescaling.
        qsim = (qr2_out + bf_out) * params["area"] / 86.4

        # Return
        next_state = {
            "s": s_out, "u_ratio": u_ratio_out, "l_ratio": l_ratio_out,
            "qr1": qr1_out, "qr2": qr2_out, "bf": bf_out
        }
        return next_state, qsim















