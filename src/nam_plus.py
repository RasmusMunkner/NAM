import jax
from jax import numpy as jnp

from nam import condition
from parameters import NAM_Parameters, NAM_State, NAM_Observation, to_physical


def step(params: NAM_Parameters, state: NAM_State, obs: NAM_Observation) -> tuple[NAM_State, jnp.ndarray]:
    """Step the NAM_plus model forward once.

    Goal is to compte:
    (s_in, u_in, l_in, qr1_in, qr2_in, bf_in) -> (s_out, u_out, l_out, qr1_out, qr2_out, bf_out)

    """
    # Decide if precipitation is rain or snow
    rain, snow = obs.p * (obs.t > 0), obs.p * (obs.t <= 0)

    # Settle snow budget
    snowmelt = jnp.maximum(0, jnp.minimum(params.c_snow * obs.t, state.s))
    s_out = state.s + snow - snowmelt

    # Settle surface water budget
    u = state.u_ratio * params.u_max
    u_budget = u + rain + snowmelt  # total water available after rain+snowmelt
    u_budget_no_rain = u + snowmelt  # total water available, discounting rain

    e_p = jnp.minimum(u_budget, obs.epot)  # Follows excel, but I think it should include rain
    u_budget -= e_p
    e_a = jnp.minimum(state.l_ratio * obs.epot, obs.epot - e_p) # only sensible for obs.epot < state.l

    q_interflow = params.ckif * condition(state.l_ratio, params.tif) * u_budget
    q_interflow = jnp.minimum(q_interflow, u_budget - e_p)  # See above
    u_budget -= q_interflow

    excess = jnp.maximum(0, u_budget - params.u_max)
    q_overflow = params.cqof * condition(state.l_ratio, params.tof) * excess
    u_ratio_out = (u_budget - excess) / params.u_max

    # Settle lower zone budget
    l_budget = (state.l_ratio * params.l_max - e_a) + (excess - q_overflow)
    percolation = (excess - q_overflow) * condition(state.l_ratio, params.tg)
    l_budget -= percolation
    l_excess = jnp.maximum(0, l_budget - params.l_max)
    q_overflow += l_excess
    l_budget -= l_excess
    l_ratio_out = l_budget / params.l_max

    # Calculate flows
    qr1_out = state.qr1 * params.ck1 + (q_overflow + q_interflow) * (1 - params.ck1)
    qr2_out = state.qr2 * params.ck2 + qr1_out * (1 - params.ck2)
    bf_out = state.bf * params.ckbf + percolation * (1 - params.ckbf) * params.c_area

    # Calculate simulated discharge
    qsim = (qr2_out + bf_out) * params.area / 86.4

    # Return
    return NAM_State(s_out, u_ratio_out, l_ratio_out, qr1_out, qr2_out, bf_out), qsim