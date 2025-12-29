from jax import numpy as jnp
from matplotlib import pyplot as plt
from nam_classic import nam_plus
from nam_classic import excel_with_defaults, example_state, example_params, example_observations

final_state, pred = nam_plus.predict_debug(example_params, example_state, example_observations)