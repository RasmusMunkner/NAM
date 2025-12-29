import jax
from jax import numpy as jnp

from nam_classic.utils import condition
from nam_classic.parameters import NAM_Parameters, NAM_State, NAM_Observation, to_physical


def step(params: NAM_Parameters, state: NAM_State, obs: NAM_Observation) -> tuple[NAM_State, jnp.ndarray]:
    """Step the NAM_plus model forward once.

    Goal is to compute:
    (s_in, u_in, l_in, qr1_in, qr2_in, bf_in) -> (s_out, u_out, l_out, qr1_out, qr2_out, bf_out)

    Compared to nam_excel, this function has several benefits:
    - handles snow/rain distribution consistently at exactly 0 degrees.
    - evaporation and interflow always depend on all available water at a given timestep
    - excess surface water is routed as overflow rather than infiltrating the lower zone and breaking the max value

    """
    # Decide if precipitation is rain or snow
    rain, snow = obs.p * (obs.t > 0), obs.p * (obs.t <= 0)

    # Settle snow budget
    snowmelt = jnp.maximum(0, jnp.minimum(params.c_snow * obs.t, state.s))
    s_out = state.s + snow - snowmelt

    # Settle surface water budget
    u = state.u_ratio * params.u_max
    u_budget = u + rain + snowmelt  # total water available after rain+snowmelt

    e_p = jnp.minimum(u_budget, obs.epot)
    u_budget -= e_p

    q_interflow = params.ckif * condition(state.l_ratio, params.tif) * u_budget
    u_budget -= q_interflow

    excess = jnp.maximum(0, u_budget - params.u_max)
    u_ratio_out = (u_budget - excess) / params.u_max

    # Settle lower zone budget
    q_overflow = params.cqof * condition(state.l_ratio, params.tof) * excess
    percolation = condition(state.l_ratio, params.tg) * (excess - q_overflow)
    l_budget = state.l_ratio * params.l_max + (excess - q_overflow - percolation)
    e_a = jnp.minimum(jnp.minimum(state.l_ratio * obs.epot, obs.epot - e_p), l_budget)
    l_budget -= e_a
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


def predict(
    params_trainable: dict[str, jnp.ndarray],
    state_trainable: dict[str, jnp.ndarray],
    params_fixed: dict[str, jnp.ndarray],
    state_fixed: dict[str, jnp.ndarray],
    obs: NAM_Observation
) -> jnp.ndarray:
    """Compute the loss for the NAM model.
    
    Notes:
    ----------
        The trainable parameters are expected to be provided in unconstrained space.
    """
    params_train = to_physical(params_trainable)
    params_fix   = params_fixed
    state_train  = to_physical(state_trainable)
    state_fix    = state_fixed

    params = NAM_Parameters(**{**params_fix, **params_train})
    state  = NAM_State(**{**state_fix, **state_train})
    
    def scan_step(state, obs_t):
        state, qsim = step(params, state, obs_t)
        return state, qsim

    obs_seq = NAM_Observation(obs.p, obs.epot, obs.t)

    final_state, qsim = jax.lax.scan(
        scan_step,
        state,
        obs_seq
    )
    return qsim


def predict_debug(params: NAM_Parameters, state: NAM_State, obs: NAM_Observation) -> tuple[NAM_State, jnp.ndarray]:
    qq = []
    for i in range(len(obs.p)):
        state, q = step(params, state, NAM_Observation(obs.p[i], obs.epot[i], obs.t[i]))
        qq.append(q)
    return state, jnp.asarray(qq)


def mse(
    params_trainable: dict[str, jnp.ndarray],
    state_trainable: dict[str, jnp.ndarray],
    params_fixed: dict[str, jnp.ndarray],
    state_fixed: dict[str, jnp.ndarray],
    obs: NAM_Observation,
    target: jnp.ndarray,
) -> jnp.ndarray:
    pred = predict(params_trainable,state_trainable, params_fixed, state_fixed, obs)
    return jnp.mean(jnp.square(pred - target))