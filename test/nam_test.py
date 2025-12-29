import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import jax
jax.config.update('jax_enable_x64', True) # controls the floating point precision
from nam_classic import *

data = pd.read_csv("nam_excel_default_results.csv", delimiter=";", decimal=",")
data["date"] = pd.to_datetime(data["date"], dayfirst=True)

initial_state = jax.tree.map(jnp.asarray, {"s": 0.,"u": 5, "l": 100., "qr1": 0.43, "qr2": 0, "bf": 0.86})
initial_params = jax.tree.map(jnp.asarray, {"area": 1055., "c_area": 0.9, "c_snow": 2., "l_max": 100., "u_max": 5., "cqof": 0.3, "ckif": 20., "tif": 0.5, "tof": 0.2, "tg": 0.5, "ck1": np.exp(-1/2), "ck2": 0., "ckbf": np.exp(-1/500)})
initial_state = NAM_State(**initial_state)
initial_params = NAM_Parameters(**initial_params)

observations = NAM_Observation(**jax.tree.map(jnp.asarray, {"p": data["p"], "epot": data["epot"], "t": data["temp"]}))
targets = jnp.asarray(data["qobs"])


states, r = predict(obs=observations, state=initial_state, params=initial_params)
r = np.asarray(r)
# states_df = pd.DataFrame.from_records([s._asdict() for s in states])[1:]

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12,12))
axf = axs.flatten()

axf[0].plot(data["date"], r-data["qsim"].to_numpy())
axf[0].set_ylabel("$\\Delta Q$")

# axf[1].plot(data["date"], states_df["s"].to_numpy()-data["ss"].to_numpy())
# axf[1].set_ylabel("$\\Delta S$")

# axf[2].plot(data["date"], states_df["u"].to_numpy()-data["u"].to_numpy())
# axf[2].set_ylabel("$\\Delta U$")

# axf[3].plot(data["date"], states_df["l"].to_numpy()-data["l"].to_numpy())
# axf[3].set_ylabel("$\\Delta L$")

plt.show(block=True)