import abc
from typing import NamedTuple, Callable
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from evosax.algorithms import CMA_ES
from tqdm.auto import tqdm
import warnings

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
        self.frozen = frozen

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

    def predict_nojit(self, obs: HydroObservation) -> tuple[list[NamedTuple], list[jnp.ndarray]]:
        return predict_nojit(self.params, obs, self.step)

    def squared_error(self, obs: HydroObservation, target: jnp.ndarray, reduce: bool = True) -> jnp.ndarray:
        return squared_error(self.params, obs, target, self.step, reduce)
    
    def nash_sutcliffe(self, obs: HydroObservation, target: jnp.ndarray):
        mse = self.squared_error(obs, target)
        mse_base = jnp.mean(jnp.square(target-jnp.mean(target)))
        return (mse_base-mse)/mse_base

    def cumulative_waterbalance(self, obs: HydroObservation, target: jnp.ndarray, relative: bool=True):
        """Relative cumulative water balance."""
        _, pred = self.predict(obs)
        csum_pred = jnp.cumsum(jnp.ravel(pred))
        csum_target = jnp.cumsum(jnp.ravel(target))
        if relative:
            return (csum_pred-csum_target)/csum_target
        else:
            return csum_pred-csum_target

    def squared_error_grad(self, obs: HydroObservation, target: jnp.ndarray) -> jnp.ndarray:
        return squared_error_grad(self.params, obs, target, self.step)

    def optimize(
            self,
            obs: HydroObservation,
            target: jnp.ndarray,
            steps: int,
            progbar: bool = True,
            optimizer: optax.GradientTransformationExtraArgs = None,
    ):
        if optimizer is None:
            optimizer = optax.sgd(learning_rate=1e-2)

        if self.frozen is not None:
            optimizer = optax.chain(optimizer, optax.transforms.freeze(self.frozen))
        opt_state = optimizer.init(self.params)

        trace = []

        for _ in tqdm(range(steps), total=steps, desc="Optimizing...", disable=not progbar):

            loss = self.squared_error(obs, target, reduce=False)
            trace.append({"loss": loss, **self.params._asdict()})

            grads = self.squared_error_grad(obs, target)
            updates, opt_state = optimizer.update(grads, opt_state)
            self.params = optax.apply_updates(self.params, updates)

        trace = {
            k: jnp.stack([trace[i][k] for i in range(len(trace))], axis=0)
            for k in trace[0].keys()
        }

        return trace

    def optimize_cma_es(
            self,
            obs: HydroObservation,
            target: jnp.ndarray,
            n_generations: int = 64,
            n_population: int = 1024,
            progbar: bool = True,
            key: jax.Array = None
    ):
        if key is None:
            key = random.PRNGKey(0)
        dummy_solution = {k:v for k,v in self.params._asdict().items() if not self.frozen._asdict()[k]}
        es = CMA_ES(population_size=n_population, solution=dummy_solution)

        es_params = es.default_params
        key, subkey = random.split(key)
        es_state = es.init(subkey, dummy_solution, es_params)

        metrics_logs = []

        for _ in tqdm(range(n_generations), desc="Optimizing...", total=n_generations, disable=not progbar):
            key, subkey = random.split(key)
            key_ask, key_eval, key_tell = random.split(subkey, 3)

            population, es_state = es.ask(key_ask, es_state, es_params)

            population_flat = jax.tree.map(jnp.ravel, population)
            params_flat = jax.tree.map(lambda x: jnp.ones(shape=(n_population,))*x, self.params)

            score = squared_error_masked(population_flat, params_flat, obs, target, self.step, reduce=False)

            es_state, metrics = es.tell(key_tell, population, score, es_state, es_params)

            metrics_logs.append(metrics)

        best_params = self.params._asdict()
        best_params.update(
            jax.tree.unflatten(jax.tree.structure(dummy_solution), metrics_logs[-1]["best_solution"])
        )
        for k in best_params:
            best_params[k] = jnp.ravel(best_params[k])
        best_params = self.params.__class__(**best_params)

        return best_params, metrics_logs






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


def predict_nojit(
        params: NamedTuple,
        obs: HydroObservation,
        step_fn: Callable[
                    [NamedTuple, HydroObservation],
                    tuple[NamedTuple, jnp.ndarray]
                ]
):
    states, preds = [], []
    s = params
    for i in range(len(obs.p)):
        s,p = step_fn(s, HydroObservation(p=obs.p[i], epot=obs.epot[i], t=obs.t[i]))
        states.append(s)
        preds.append(p)

    return states, preds


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


def squared_error_masked(
        params,
        params_mask: NamedTuple,
        obs: HydroObservation,
        target: jnp.ndarray,
        step_fn: Callable[
            [NamedTuple, HydroObservation],
            tuple[NamedTuple, jnp.ndarray]
        ],
        reduce: bool = False
):
    base_params = params_mask._asdict()
    base_params.update(params)
    base_params = params_mask.__class__(**base_params)
    return squared_error(base_params, obs, target, step_fn, reduce)

squared_error_grad = jax.grad(squared_error, argnums=0)

# squared_error_grad_vmap = jax.vmap(squared_error_grad, in_axes=[0, 0, 0, 0, None, None, None])







