import jax
from jax import numpy as jnp
from nam_classic import nam_excel_hydroapi
import hydro_model_interface
import data
jax.config.update("jax_enable_x64", True)

params = nam_excel_hydroapi.NAMParametersMock.from_physical()
model = nam_excel_hydroapi.NAM(params)

# pred = model.predict_nojit(data.observations)
for i in range(1000):
    obs = hydro_model_interface.HydroObservation(
            p=data.observations.p[i],
            epot=data.observations.epot[i],
            t=data.observations.t[i]
        )
    temp, qsim = model.step(
        params,
        obs
    )
    target = data.nam_excel_results["qsim"].iloc[i]
    if jnp.abs(qsim-target) > 1e-6:
        pass
        model.step(params, obs)
    params = temp
