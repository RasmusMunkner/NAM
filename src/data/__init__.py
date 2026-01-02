from importlib.resources import files
import pandas as pd
from jax import numpy as jnp

from hydro_model_interface import HydroObservation

# Load timeseries for excel reference run
_excel_result_path = files("data").joinpath("raw/nam_excel_default_results.csv")
nam_excel_results = pd.read_csv(_excel_result_path, delimiter=";", decimal=",")
nam_excel_results["date"] = pd.to_datetime(nam_excel_results["date"], dayfirst=True)
observations_uncorrected = HydroObservation(
    p = jnp.array(nam_excel_results["p"]),
    epot = jnp.array(nam_excel_results["epot"]),
    t = jnp.array(nam_excel_results["temp"]),
)
discharge_uncorrected = jnp.array(nam_excel_results["qobs"]).reshape(-1,1)

# Load corrected timeseries for the Allergaarde catchment
_timeseries_path = files("data").joinpath("processed/timeseries_corrected.csv")
timeseries = pd.read_csv(_timeseries_path)
timeseries["date"] = pd.to_datetime(timeseries["date"])
timeseries = timeseries[~pd.isnull(timeseries["discharge"])]
timeseries = timeseries.merge(nam_excel_results[["date", "epot"]], on="date")
observations = HydroObservation(
    p = jnp.array(timeseries["precipitation"]),
    epot = jnp.array(timeseries["epot"]),
    t = jnp.array(timeseries["temperature"]),
)
discharge = jnp.array(timeseries["discharge"]).reshape(-1,1)

# Compute observations converted to the same unit as discharge (m3/s)
observations_m3s = HydroObservation(
    p = jnp.array(timeseries["precipitation"]) * 1055 / 86.4,
    epot = jnp.array(timeseries["epot"]) * 1055 / 86.4,
    t = jnp.array(timeseries["temperature"]),
)
