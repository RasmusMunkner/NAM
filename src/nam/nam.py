import jax
from jax import numpy as jnp

from nam.parameters import NAM_Parameters, NAM_State, NAM_Observation, to_physical
    
def step(params: NAM_Parameters, state: NAM_State, obs: NAM_Observation) -> tuple[NAM_State, jnp.ndarray]:
    """Step the NAM model forward once.
    
    Goal is to compute:
    (s_in, u_in, l_in, qr1_in, qr2_in, bf_in) -> (s_out, u_out, l_out, qr1_out, qr2_out, bf_out)
    
    Note, there is a mistake in the original NAM excel sheet.
    There, snowmelt happens only for t < 0, and rain happens only for t > 0.
    I.e. when t=0, we lose water.
    
    """
    # Decide if precipitation is rain or snow
    rain, snow = obs.p * (obs.t > 0), obs.p * (obs.t < 0) # Replicates excel, but likely a mistake with no weak inequality
    
    # Settle snow budget
    snowmelt = jnp.maximum(0, jnp.minimum(params.c_snow*obs.t, state.s))
    s_out = state.s + snow - snowmelt
    
    # Settle surface water budget
    u = state.u_ratio * params.u_max
    u_budget = u + rain + snowmelt # total water available after rain+snowmelt
    u_budget_no_rain = u + snowmelt # total water available, discounting rain
    
    e_p = jnp.minimum(u_budget_no_rain, obs.epot) # Follows excel, but I think it should include rain
    u_budget -= e_p
    e_a = jnp.minimum(state.l_ratio*obs.epot, obs.epot-e_p) # min of (estimate, energy constraint, (mass constraint???, missing???))
    
    q_interflow = params.ckif * condition(state.l_ratio, params.tif) * u # Follows excel, but why not just use u_budget?
    q_interflow = jnp.minimum(q_interflow, u_budget_no_rain - e_p) # See above
    u_budget -= q_interflow
    
    excess = jnp.maximum(0, u_budget - params.u_max)
    q_overflow = params.cqof * condition(state.l_ratio, params.tof) * excess
    u_ratio_out = (u_budget - excess) / params.u_max
    
    # Settle lower zone budget
    percolation = (excess - q_overflow) * condition(state.l_ratio, params.tg)
    dl = excess - q_overflow - percolation
    l_ratio_out = state.l_ratio + (dl - e_a) / params.l_max
    
    # Calculate flows
    qr1_out = state.qr1 * params.ck1 + (q_overflow + q_interflow) * (1-params.ck1)
    qr2_out = state.qr2 * params.ck2 + qr1_out * (1-params.ck2)
    bf_out = state.bf * params.ckbf + percolation * (1-params.ckbf) * params.c_area
    
    # Calculate simulated discharge
    qsim = (qr2_out + bf_out) * params.area / 86.4 # params.area rescaled from mm/d to m3/d. In the excel sheet, they also divide by 86.4 for some reason
    
    # Return
    return NAM_State(s_out, u_ratio_out, l_ratio_out, qr1_out, qr2_out, bf_out), qsim


def condition(saturation: jnp.ndarray, threshold: jnp.ndarray) -> jnp.ndarray:
    return (saturation - threshold) / (1 - threshold) * (saturation > threshold)


def predict(params: NAM_Parameters, state: NAM_State, obs: NAM_Observation) -> tuple[NAM_State, jnp.ndarray]:
    def scan_step(state, obs_t):
        state, qsim = step(params, state, obs_t)
        return state, qsim

    obs_seq = NAM_Observation(obs.p, obs.epot, obs.t)

    final_state, qsim = jax.lax.scan(
        scan_step,
        state,
        obs_seq
    )
    return final_state, qsim


def mse(
    params_trainable: dict[str, jnp.ndarray],
    state_trainable: dict[str, jnp.ndarray],
    params_fixed: dict[str, jnp.ndarray],
    state_fixed: dict[str, jnp.ndarray],
    obs: NAM_Observation,
    target: jnp.ndarray,
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

    _, pred = predict(params, state, obs)
    return jnp.mean(jnp.square(pred - target))


    
    

    
    
    
    
    

    
    
    
    







