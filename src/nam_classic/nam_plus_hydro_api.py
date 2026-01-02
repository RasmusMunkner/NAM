from jax import numpy as jnp
from hydro_model_interface import HydroModel, HydroObservation
from hydro_model_interface.parameters import inv_softplus, inv_sigmoid
from nam_classic.nam_excel_hydroapi import NAMParameters, NAM
from nam_classic.utils import condition

class NAMPlus(NAM):
    """Exact replication of the NAM model as it was implemented in excel."""


    @staticmethod
    def step(
            params: NAMParameters,
            obs: HydroObservation
    ):

        # Decide if precipitation is rain or snow
        rain, snow = obs.p * (obs.t > 0), obs.p * (obs.t <= 0)

        # Settle snow budget
        snowmelt = jnp.maximum(0, jnp.minimum(params.c_snow * obs.t, params.s))
        s_out = params.s + snow - snowmelt

        # Settle surface water budget
        u = params.u_ratio * params.u_max
        u_budget = u + rain + snowmelt  # total water available after rain+snowmelt

        e_p = jnp.minimum(u_budget, obs.epot)
        u_budget -= e_p

        q_interflow = params.ckif * condition(params.l_ratio, params.tif) * u_budget
        u_budget -= q_interflow

        excess = jnp.maximum(0, u_budget - params.u_max)
        u_ratio_out = (u_budget - excess) / params.u_max

        # Settle lower zone budget
        q_overflow = params.cqof * condition(params.l_ratio, params.tof) * excess
        percolation = condition(params.l_ratio, params.tg) * (excess - q_overflow)
        l_budget = params.l_ratio * params.l_max + (excess - q_overflow - percolation)
        e_a = jnp.minimum(jnp.minimum(params.l_ratio * obs.epot, obs.epot - e_p), l_budget)
        l_budget -= e_a
        l_excess = jnp.maximum(0, l_budget - params.l_max)
        q_overflow += l_excess
        l_budget -= l_excess
        l_ratio_out = l_budget / params.l_max

        # Calculate flows
        qr1_out = params.qr1 * params.ck1 + (q_overflow + q_interflow) * (1 - params.ck1)
        qr2_out = params.qr2 * params.ck2 + qr1_out * (1 - params.ck2)
        bf_out = params.bf * params.ckbf + percolation * (1 - params.ckbf) * params.c_area

        # Calculate simulated discharge
        qsim = (qr2_out + bf_out) * params.area / 86.4

        # Return
        next_state = params._asdict()
        next_state.update({
            "s_": inv_softplus(s_out), "u_ratio_": inv_sigmoid(u_ratio_out), "l_ratio_": inv_sigmoid(l_ratio_out),
            "qr1_": inv_sigmoid(qr1_out), "qr2_": inv_sigmoid(qr2_out), "bf_": inv_sigmoid(bf_out)
        })
        return NAMParameters(**next_state), qsim



































