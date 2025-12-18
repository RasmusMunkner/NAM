from importlib.resources import files
from jax import numpy as jnp
import numpy as np
import pandas as pd
import nam.nam_excel as nam
from nam.parameters import NAM_Parameters, NAM_Observation, NAM_State, pack_params

example_state = NAM_State(**pack_params(s=0, qr2=0, u_ratio=1., l_ratio=1., qr1=0.43, bf=0.86))
example_params = NAM_Parameters(**pack_params(
    area=1055, c_area=0.9, c_snow=2, ck2=0., l_max=100., u_max=5., cqof=0.3, ckif=1/20, tif=0.5, tof=0.2, tg=0.5, ck1=np.exp(-1/2), ckbf=np.exp(-1/500)
))

# Reference data for testing
path_to_excel_reference = files("nam.data").joinpath("excel_with_defaults.csv")
excel_with_defaults = pd.read_csv(path_to_excel_reference, delimiter=";", decimal=",")
excel_with_defaults["date"] = pd.to_datetime(excel_with_defaults["date"], dayfirst=True)
example_observations = NAM_Observation(
    p=jnp.asarray(excel_with_defaults["p"]),
    epot=jnp.asarray(excel_with_defaults["epot"]),
    t=jnp.asarray(excel_with_defaults["temp"]),
)


