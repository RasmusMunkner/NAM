from jax import numpy as jnp

from hydrosystem.discharge.nam.parameters import NAMParameters
from hydrosystem.discharge.nam.observation import NAMObservation, NAMTarget
from hydrosystem.discharge.nam.math import threshold_linear


def step_excel(
        params: NAMParameters,
        obs: NAMObservation
) -> tuple[NAMParameters, NAMTarget]:
    """One step of NAM model calculation.

    Notes:
        This function faithfully replicates the excel implementation.
    """
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
    e_a = jnp.minimum(params.l_ratio * obs.epot, obs.epot - e_p)

    # Follows excel, but why not just use u_budget?
    q_interflow = params.ckif * threshold_linear(params.l_ratio, params.tif) * u
    q_interflow = jnp.minimum(q_interflow, u_budget_no_rain - e_p)
    u_budget -= q_interflow

    excess = jnp.maximum(0, u_budget - params.u_max)
    q_overflow = params.cqof * threshold_linear(params.l_ratio, params.tof) * excess
    u_ratio_out = (u_budget - excess) / params.u_max

    # Settle lower zone budget
    percolation = (excess - q_overflow) * threshold_linear(params.l_ratio, params.tg)
    dl = excess - q_overflow - percolation
    l_ratio_out = params.l_ratio + (dl - e_a) / params.l_max

    # Calculate flows
    qr1_out = params.qr1 * params.ck1 + (q_overflow + q_interflow) * (1 - params.ck1)
    qr2_out = params.qr2 * params.ck2 + qr1_out * (1 - params.ck2)
    bf_out = params.bf * params.ckbf + percolation * (1 - params.ckbf) * params.c_area

    # Calculate simulated discharge
    # params.area rescaled from mm/d to m3/s. Note the /86.4 handles the time rescaling.
    qsim = (qr2_out + bf_out)# * params.area / 86.4

    # Return
    updates = {
        "s": s_out, "u_ratio": u_ratio_out, "l_ratio": l_ratio_out,
        "qr1": qr1_out, "qr2": qr2_out, "bf": bf_out
    }
    next_state = params.update(updates)
    target = NAMTarget(
        q=qsim, eact=e_p + e_a, perc=percolation
    )
    return next_state, target


def step(
        params: NAMParameters,
        obs: NAMObservation
) -> tuple[NAMParameters, NAMTarget]:
    """One step of the NAM model calculation."""

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

    q_interflow = params.ckif * threshold_linear(params.l_ratio, params.tif) * u_budget
    u_budget -= q_interflow

    excess = jnp.maximum(0, u_budget - params.u_max)
    u_ratio_out = (u_budget - excess) / params.u_max

    # Settle lower zone budget
    q_overflow = params.cqof * threshold_linear(params.l_ratio, params.tof) * excess
    percolation = threshold_linear(params.l_ratio, params.tg) * (excess - q_overflow)
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
    qsim = (qr2_out + bf_out)# * params.area / 86.4

    # Return
    updates = {
        "s": s_out, "u_ratio": u_ratio_out, "l_ratio": l_ratio_out,
        "qr1": qr1_out, "qr2": qr2_out, "bf": bf_out
    }
    next_state = params.update(updates)
    target = NAMTarget(
        q=qsim, eact=e_p+e_a, perc=percolation
    )
    return next_state, target














