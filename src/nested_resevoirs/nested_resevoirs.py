import jax
from jax import numpy as jnp
from nested_resevoirs.parameters import NRParameters, NRState
from data import Observation

def step(params: NRParameters, state: NRState, obs: Observation) -> tuple[NRState, jnp.ndarray]:

    # Step 0: Initialize budgets
    snow_budget = state.snow
    surface_budget = state.surface
    subsurface_budget = state.subsurface
    groundwater_budget = state.groundwater
    flow_budget = jnp.array([0])

    # Step 1: Settle precipitation
    snow_budget += obs.p * (obs.t <= 0)
    surface_budget += obs.p * (obs.t > 0)

    # Step 2: Settle snowmelt
    snowmelt = params.snowmelt(jnp.clip(obs.t / 10, 0, 1)) * snow_budget
    snow_budget -= snowmelt
    surface_budget += snowmelt

    # Step 3: Settle evaporation
    # Step 3.1. Evaporation
    evaporation = jnp.minimum(obs.epot, surface_budget)
    surface_budget -= evaporation
    # Step 3.2: Transpiration
    transpiration = jnp.minimum(obs.epot - evaporation, subsurface_budget)
    subsurface_budget -= transpiration

    # Step 4: Settle percolation
    # Step 4.1: Subsurface -> Groundwater
    g_sat = groundwater_budget / params.groundwater_storage
    l_sat = subsurface_budget / params.subsurface_storage
    l_percolation_ideal = params.subsurface_percolation(0.5 * (l_sat - g_sat) + 0.5) * subsurface_budget
    l_percolation = jnp.minimum(l_percolation_ideal, params.groundwater_storage - groundwater_budget)
    groundwater_budget += l_percolation
    subsurface_budget -= l_percolation

    # Step 4.2 Surface -> Subsurface
    s_sat = jnp.clip(surface_budget / params.surface_storage, 0, 1)
    l_sat = subsurface_budget / params.subsurface_storage
    s_percolation_ideal = params.surface_percolation(0.5*(s_sat - l_sat) + 0.5) * surface_budget
    s_percolation = jnp.minimum(s_percolation_ideal, params.subsurface_storage - subsurface_budget)
    subsurface_budget += s_percolation
    surface_budget -= s_percolation

    # Step 5: Allocate flow
    s_sat = jnp.clip(surface_budget / params.surface_storage, 0, 1)
    l_sat = subsurface_budget / params.subsurface_storage
    g_sat = groundwater_budget / params.groundwater_storage

    overflow = params.surface_flow(s_sat)*surface_budget
    overflow = jnp.maximum(overflow, jnp.maximum(surface_budget - params.surface_storage, 0))
    flow_budget += overflow
    surface_budget -= overflow

    interflow = params.subsurface_flow(l_sat) * subsurface_budget
    flow_budget += interflow
    subsurface_budget -= interflow

    baseflow = params.groundwater_flow(g_sat) * groundwater_budget
    flow_budget += baseflow
    groundwater_budget -= baseflow

    return NRState(snow_budget, surface_budget, subsurface_budget, groundwater_budget), flow_budget


def predict(params: NRParameters, state: NRState, obs: Observation) -> tuple[NRState, jnp.ndarray]:

    def scan_step(state_t, obs_t):
        state_tp1, qsim_t = step(params, state_t, obs_t)
        return state_tp1, qsim_t

    final_state, qsim = jax.lax.scan(
        scan_step,
        state,
        obs
    )
    return final_state, qsim


def mse(params: NRParameters, state: NRState, obs: Observation, target: jnp.ndarray) -> jnp.ndarray:
    _, prediction = predict(params, state, obs)
    return jnp.mean(jnp.square(prediction - target))














