from importlib.resources import files
from jax import numpy as jnp
import numpy as np
import pandas as pd
from nam_classic.parameters import NAM_Parameters, NAM_Observation, NAM_State, pack_params

# Reference state/params for testing
default_state = NAM_State(**pack_params(s=0, qr2=0, u_ratio=1., l_ratio=1., qr1=0.43, bf=0.86))
default_params = NAM_Parameters(**pack_params(
    area=1055, c_area=0.9, c_snow=2, ck2=0., l_max=100., u_max=5., cqof=0.3, ckif=1/20, tif=0.5, tof=0.2, tg=0.5, ck1=np.exp(-1/2), ckbf=np.exp(-1/500)
))

# Reasonable parameter spaces - Take these with a grain of salt
default_state_space = {
    "s": (0,0),
    "u_ratio": (0,1),
    "l_ratio": (0,1),
    "qr1": (0, 1),
    "qr2": (0, 0),
    "bf": (0.5, 1)
}
default_params_space = {
    "area": (1055, 1055),
    "c_area": (0.9, 0.9),
    "c_snow": (2, 2),
    "l_max": (50, 500),
    "u_max": (5, 35),
    "cqof": (0, 1),
    "ckif": (1/40,1/20),
    "tif": (0, 0.9),
    "tof": (0, 0.9),
    "tg": (0, 0.9),
    "ck1": (np.exp(-10), np.exp(-1/10)),
    "ck2": (0,0),
    "ckbf": (np.exp(-1/20),np.exp(-1/700)),
}

# Reference data for testing
_path_to_excel_reference = files("nam_classic.data").joinpath("nam_excel_default_results.csv")
excel_results = pd.read_csv(_path_to_excel_reference, delimiter=";", decimal=",")
excel_results["date"] = pd.to_datetime(excel_results["date"], dayfirst=True)

default_observations = NAM_Observation(
    p=jnp.asarray(excel_results["p"]),
    epot=jnp.asarray(excel_results["epot"]),
    t=jnp.asarray(excel_results["temp"]),
)






















