import abc
from typing import NamedTuple, Callable
import jax
import jax.numpy as jnp
import optax
from tqdm.auto import tqdm

from hydro_model_interface.parameters import to_physical, to_unconstrained

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
            params_trainable: dict[str, jnp.ndarray],
            state_trainable: dict[str, jnp.ndarray],
            params_fixed: dict[str, jnp.ndarray],
            state_fixed: dict[str, jnp.ndarray],
            optimizer: optax.GradientTransformationExtraArgs = None,
    ):
        self._params_trainable = to_unconstrained(params_trainable)
        self._state_trainable = to_unconstrained(state_trainable)
        self._params_fixed = to_unconstrained(params_fixed)
        self._state_fixed = to_unconstrained(state_fixed)
        if optimizer is None:
            self.optimizer = optax.sgd(learning_rate=1e-3)
        else:
            self.optimizer = optimizer
        self.opt_state = self.optimizer.init(self._trainable)

    @property
    def params_trainable(self):
        return to_physical(self._params_trainable)

    @property
    def state_trainable(self):
        return to_physical(self._state_trainable)

    @property
    def params_fixed(self):
        return to_physical(self._params_fixed)

    @property
    def state_fixed(self):
        return to_physical(self._state_fixed)

    @property
    def params(self):
        return {**self.params_trainable, **self.params_fixed}

    @property
    def state(self):
        return {**self.state_trainable, **self.state_fixed}

    @property
    def _trainable(self):
        return {**self._params_trainable, **self._state_trainable}

    @property
    def _fixed(self):
        return {**self._params_fixed, **self._state_fixed}

    def freeze(self, keys):
        """Freeze a subset of parameters. Note this resets optimizer state."""
        for p in self._params_trainable:
            if p in keys:
                self._params_fixed[p] = self._params_trainable[p]
                del self._params_trainable[p]
        for s in self._state_trainable:
            if s in keys:
                self._state_fixed[s] = self._state_trainable[s]
                del self._state_trainable[s]
        self.opt_state = self.optimizer.init(self._trainable)

    def unfreeze(self, keys):
        """Unfreeze a subset of parameters. Note this resets optimizer state."""
        for p in self._params_trainable:
            if p in keys:
                self._params_trainable[p] = self._params_fixed[p]
                del self._params_fixed[p]
        for s in self._state_trainable:
            if s in keys:
                self._state_trainable[s] = self._state_fixed[s]
                del self._state_trainable[s]
        self.opt_state = self.optimizer.init(self._trainable)

    @staticmethod
    @abc.abstractmethod
    def step(
            params: dict[str, jnp.ndarray],
            state: dict[str, jnp.ndarray],
            obs: HydroObservation
    ) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
        """Abstract method to advance hydro model one timestep. Implemented in terms of physical parameters."""
        pass

    def predict(self, obs: HydroObservation) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
        return predict(self._params_trainable, self._state_trainable, self._params_fixed, self._state_fixed, obs, self.step)

    def squared_error(self, obs: HydroObservation, target: jnp.ndarray, reduce: bool = True) -> jnp.ndarray:
        return squared_error(
            self._params_trainable, self._state_trainable, self._params_fixed, self._state_fixed,
            obs, target, self.step, reduce
        )

    def squared_error_grad(self, obs: HydroObservation, target: jnp.ndarray) -> jnp.ndarray:
        return squared_error_grad(
            self._params_trainable, self._state_trainable, self._params_fixed, self._state_fixed, obs, target, self.step
        )

    # def squared_error_grad_vmap(self, obs: HydroObservation, target: jnp.ndarray) -> jnp.ndarray:
    #     return squared_error_grad_vmap(
    #         self._params_trainable, self._state_trainable, self._params_fixed, self._state_fixed, obs, target, self.step
    #     )

    def optimize(self, obs: HydroObservation, target: jnp.ndarray, steps: int, progbar: bool = True):
        trace = []

        for _ in tqdm(range(steps), total=steps, desc="Optimizing...", disable=not progbar):

            loss = self.squared_error(obs, target, reduce=False)
            trace.append({"loss": loss, **self.params_trainable, **self.state_trainable})

            grads = self.squared_error_grad(obs, target)
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self._params_trainable = optax.apply_updates(self._params_trainable, updates[0])
            self._state_trainable = optax.apply_updates(self._state_trainable, updates[1])

        return trace


def predict(
        params_trainable: dict[str, jnp.ndarray],
        state_trainable: dict[str, jnp.ndarray],
        params_fixed: dict[str, jnp.ndarray],
        state_fixed: dict[str, jnp.ndarray],
        obs: HydroObservation,
        step_fn: Callable[
            [dict[str, jnp.ndarray], dict[str, jnp.ndarray], HydroObservation],
            tuple[dict[str, jnp.ndarray], jnp.ndarray]
        ]
):
    params = to_physical({**params_trainable, **params_fixed})
    state = to_physical({**state_trainable, **state_fixed})

    def scan_step(state_t, obs_t):
        state_tp1, qsim_t = step_fn(params, state_t, obs_t)
        return state_tp1, qsim_t

    final_state, qsim = jax.lax.scan(
        scan_step,
        state,
        obs
    )
    return final_state, qsim


def squared_error(
        params_trainable: dict[str, jnp.ndarray],
        state_trainable: dict[str, jnp.ndarray],
        params_fixed: dict[str, jnp.ndarray],
        state_fixed: dict[str, jnp.ndarray],
        obs: HydroObservation,
        target: jnp.ndarray,
        step_fn: Callable[
            [dict[str, jnp.ndarray], dict[str, jnp.ndarray], HydroObservation],
            tuple[dict[str, jnp.ndarray], jnp.ndarray]
        ],
        reduce: bool = True
):
    _, predicted = predict(params_trainable, state_trainable, params_fixed, state_fixed, obs, step_fn)
    mse = jnp.mean(jnp.square(predicted - target), axis=0)
    if reduce:
        return jnp.nansum(mse)
    else:
        return mse

squared_error_grad = jax.grad(squared_error, argnums=[0,1])

# squared_error_grad_vmap = jax.vmap(squared_error_grad, in_axes=[0, 0, 0, 0, None, None, None])







