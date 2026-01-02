import abc
from typing import NamedTuple, Callable
import jax
import jax.numpy as jnp
import optax
from tqdm.auto import tqdm

"""Hydro API. Here are the rules/design principles:

1. API is fully jax native, but exposes a stateful wrapper class for end user convenience.
2. All parameters reside in unconstrained space, but end-users specify and retrieve these in constrained space.
3. Initial state and parameters are kept separate, but optimized similarly.
4. Parameters and state must be subclasses of NamedTuple.
5. Freezing certain parameters during optimization is achieved through masks.
"""



class HydroObservation(NamedTuple):
    """Class for holding input data to hydrological models.

    Parameters
    ------------
    p: jnp.ndarray
        Accumulated precipitation for the timestep.
    epot: jnp.ndarray
        Accumulated potential evapotranspiration for the timestep.
        Should be given in the same unit as precipitation.
    t: jnp.ndarray
        Representative temperature for the timestep.
        Should be given in degrees Celsius.
    """
    p: jnp.ndarray  # Precipitation
    epot: jnp.ndarray  # Potential Evapotranspiration
    t: jnp.ndarray  # Temperature (Celsius)


class HydroModel(abc.ABC):
    """Abstract base class for hydro model."""

    def __init__(
            self,
            params: NamedTuple,
            frozen: NamedTuple = None,
            optimizer: optax.GradientTransformationExtraArgs = None,
    ):
        self.params: NamedTuple = params
        if optimizer is None:
            self.optimizer = optax.sgd(learning_rate=1e-2)
        else:
            self.optimizer = optimizer
        if frozen is not None:
            self.optimizer = optax.chain(self.optimizer, optax.transforms.freeze(frozen))
        self.opt_state = self.optimizer.init(self.params)

    @staticmethod
    @abc.abstractmethod
    def step(
            params: NamedTuple,
            obs: HydroObservation
    ) -> tuple[NamedTuple, jnp.ndarray]:
        """Abstract method to advance hydro model one timestep. Implemented in terms of physical parameters."""
        pass

    def predict(self, obs: HydroObservation) -> tuple[NamedTuple, jnp.ndarray]:
        return predict(self.params, obs, self.step)

    def squared_error(self, obs: HydroObservation, target: jnp.ndarray, reduce: bool = True) -> jnp.ndarray:
        return squared_error(self.params, obs, target, self.step, reduce)

    def squared_error_grad(self, obs: HydroObservation, target: jnp.ndarray) -> jnp.ndarray:
        return squared_error_grad(self.params, obs, target, self.step)

    def optimize(self, obs: HydroObservation, target: jnp.ndarray, steps: int, progbar: bool = True):
        trace = []

        for _ in tqdm(range(steps), total=steps, desc="Optimizing...", disable=not progbar):

            loss = self.squared_error(obs, target, reduce=False)
            trace.append({"loss": loss, **self.params._asdict()})

            grads = self.squared_error_grad(obs, target)
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.params = optax.apply_updates(self.params, updates)

        return trace


def predict(
        params: NamedTuple,
        obs: HydroObservation,
        step_fn: Callable[
            [NamedTuple, HydroObservation],
            tuple[NamedTuple, jnp.ndarray]
        ]
):

    def scan_step(params_t, obs_t):
        params_tp1, qsim_t = step_fn(params_t, obs_t)
        return params_tp1, qsim_t

    final_state, qsim = jax.lax.scan(
        scan_step,
        params,
        obs
    )
    return final_state, qsim


def squared_error(
        params: NamedTuple,
        obs: HydroObservation,
        target: jnp.ndarray,
        step_fn: Callable[
            [NamedTuple, HydroObservation],
            tuple[NamedTuple, jnp.ndarray]
        ],
        reduce: bool = True
):
    _, predicted = predict(params, obs, step_fn)
    mse = jnp.mean(jnp.square(predicted - target), axis=0)
    if reduce:
        return jnp.nansum(mse)
    else:
        return mse

squared_error_grad = jax.grad(squared_error, argnums=0)

# squared_error_grad_vmap = jax.vmap(squared_error_grad, in_axes=[0, 0, 0, 0, None, None, None])







