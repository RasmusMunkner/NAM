from pathlib import Path
import numpy as np
import jax.numpy as jnp
import pandas as pd
path_to_data = Path("../src/data/processed")

from hydrosystem.discharge import nam
from hydrosystem.discharge.nam.observation import NAMTarget
from hydrosystem.discharge.nam.step import step, step_excel

from matplotlib import pyplot as plt

timeseries = pd.read_csv(path_to_data/ "timeseries_corrected.csv")
timeseries_train = timeseries[
    np.logical_and.reduce((
        pd.to_datetime(timeseries["date"]).dt.year >= 1971, # Not part of reference period
        pd.to_datetime(timeseries["date"]).dt.year <= 1990, # Not part of reference period
    ))
]
timeseries_validation = timeseries[
    np.logical_and.reduce((
        pd.to_datetime(timeseries["date"]).dt.year >= 1991, # Not part of reference period
        pd.to_datetime(timeseries["date"]).dt.year <= 2000, # Not part of reference period
    ))
]
observations_train = nam.NAMObservation(
    p=jnp.asarray(timeseries_train["p"]),
    t=jnp.asarray(timeseries_train["t"]),
    epot=jnp.asarray(timeseries_train["epot"])
)
target_train = NAMTarget.from_partial(q=timeseries_train["q"])
observations_validation = nam.NAMObservation(
    p=jnp.asarray(timeseries_validation["p"]),
    t=jnp.asarray(timeseries_validation["t"]),
    epot=jnp.asarray(timeseries_validation["epot"])
)
target_validation = NAMTarget.from_partial(timeseries_validation["q"])

initial_params = nam.NAMParameters.from_physical()

step_excel(initial_params, nam.NAMObservation(p=observations_train.p[0], t=observations_train.t[0], epot=observations_train.epot[0]))
step(initial_params, nam.NAMObservation(p=observations_train.p[0], t=observations_train.t[0], epot=observations_train.epot[0]))

initial_params_final, qsim_initial_params = nam.predict(initial_params, observations_train)

optimized_params, optimizer_trace = nam.optimize_cma_es(initial_params, observations_train, target_train)