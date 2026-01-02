import jax
from jax import numpy as jnp
from hydro_model_interface import HydroModel, HydroObservation
from hydro_model_interface.parameters import inv_sigmoid, inv_softplus

from nested_resevoirs.parameters import NRParameters

class NestedResevoirs(HydroModel):
    """Implementation of the Nested Resevoirs model."""

    @staticmethod
    def step(
            params: NRParameters,
            obs: HydroObservation
    ) -> tuple[NRParameters, jnp.ndarray]:

        # Step 0: Initialize budgets
        snow_budget = params.snow_amount
        surface_budget = params.surface_amount
        subsurface_budget = params.subsurface_amount
        groundwater_budget = params.groundwater_amount
        flow_budget = jnp.array([0])

        # Step 1: Settle precipitation
        snow_budget += obs.p * (obs.t <= 0)
        surface_budget += obs.p * (obs.t > 0)

        # Step 2: Settle snowmelt
        snowmelt = params.snowmelt.parmap(jnp.clip(obs.t / 10, 0, 1))*snow_budget
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

        l_percolation_ideal = params.subsurface_percolation.pmap(0.5 * (l_sat - g_sat) + 0.5) * subsurface_budget
        l_percolation = jnp.minimum(l_percolation_ideal, params.groundwater_storage - groundwater_budget)
        groundwater_budget += l_percolation
        subsurface_budget -= l_percolation

        # Step 4.2 Surface -> Subsurface
        s_sat = jnp.clip(surface_budget / params.surface_storage, 0, 1)
        l_sat = subsurface_budget / params.subsurface_storage
        s_percolation_ideal = params.surface_percolation.pmap(0.5 * (s_sat - l_sat) + 0.5) * surface_budget
        s_percolation = jnp.minimum(s_percolation_ideal, params.subsurface_storage - subsurface_budget)
        subsurface_budget += s_percolation
        surface_budget -= s_percolation

        # Step 5: Allocate flow
        s_sat = jnp.clip(surface_budget / params.subsurface_storage, 0, 1)
        l_sat = subsurface_budget / params.subsurface_storage
        g_sat = groundwater_budget / params.groundwater_storage

        overflow = params.surface_flow.pmap(s_sat) * surface_budget
        overflow = jnp.maximum(overflow, jnp.maximum(surface_budget - params.surface_storage, 0))
        flow_budget += overflow
        surface_budget -= overflow

        interflow = params.subsurface_flow.pmap(l_sat) * subsurface_budget
        flow_budget += interflow
        subsurface_budget -= interflow

        baseflow = params.groundwater_flow.pmap(g_sat) * groundwater_budget
        flow_budget += baseflow
        groundwater_budget -= baseflow

        # Step 6: Return
        next_state = params._asdict()
        next_state.update({
            "snow_amount_": inv_softplus(snow_budget),
            "surface_ratio_": inv_sigmoid(surface_budget / params.surface_storage),
            "subsurface_ratio_": inv_sigmoid(subsurface_budget / params.subsurface_storage),
            "groundwater_ratio_": inv_sigmoid(groundwater_budget / params.groundwater_storage),
        })
        next_state = NRParameters(**next_state)
        return next_state, flow_budget

















































