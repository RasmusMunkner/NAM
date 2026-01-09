import jax
import jax.numpy as jnp
from jax import random

from evosax.algorithms import CMA_ES

from hydrosystem.discharge.nam.observation import NAMTarget

from typing import Callable
from tqdm.auto import tqdm

from hydrosystem.discharge.nam.parameters import NAMParameters
from hydrosystem.discharge.nam.observation import NAMObservation, NAMTarget
from hydrosystem.discharge.nam.step import step


def predict(
        params: NAMParameters,
        obs: NAMObservation,
        step_fn: Callable[
            [NAMParameters, NAMObservation],
            tuple[NAMParameters, NAMTarget]
        ] = step
) -> tuple[NAMParameters, NAMTarget]:
    """Predict discharge from a set of NAMParameters and a NAMObservation time series."""
    obs = obs.canonical()

    def scan_step(params_t, obs_t):
        params_tp1, qsim_t = step_fn(params_t, obs_t)
        return params_tp1, qsim_t

    final_state, pred = jax.lax.scan(
        scan_step,
        params,
        obs
    )
    return final_state, pred


def optimize_cma_es(
        params: NAMParameters,
        obs: NAMObservation,
        target: NAMTarget,
        weights: NAMTarget = None,
        pinball_alpha: float = None,
        abssum_weight: float = 0,
        step_fn: Callable[
            [NAMParameters, NAMObservation],
            tuple[NAMParameters, NAMTarget]
        ] = step,
        n_generations: int = 32,
        n_population: int = 256,
        params_mask: NAMParameters = None,
        key: jax.Array = None,
        progbar: bool = True
):
    """Optimize NAMParameters using the CMA-ES algorithm."""

    prediction_shape = target.q.shape
    obs_shape_ok = jax.tree.all(jax.tree.map(lambda x: x.shape == prediction_shape, obs))
    target_shape_ok = jax.tree.all(jax.tree.map(lambda x: x.shape == prediction_shape, target))
    if not obs_shape_ok or not target_shape_ok:
        raise ValueError("Shape of observations and targets do not match.")
    if weights is None:
        weights = jax.tree.map(lambda x: jnp.ones_like(x), target)
    elif not jax.tree.all(jax.tree.map(lambda x: x.shape == prediction_shape, weights)):
        raise ValueError("Shape of weights and target do not match.")

    if key is None:
        key = random.PRNGKey(0)

    all_1d = jax.tree.all(jax.tree.map(jnp.isscalar, params))
    if not all_1d:
        raise ValueError("Initial parameter values must all be scalar.")

    if params_mask is None:
        params_mask = params.default_freezemask()
    params_dict = params._asdict()
    params_mask_dict = params_mask._asdict()
    optimizable_params_dict = {k:v for k,v in params_dict.items() if not params_mask_dict[k]}
    frozen_params_dict = {k:v for k,v in params_dict.items() if params_mask_dict[k]}
    
    def complete_params(partial_params):
        completed_params = {}
        for k in params_dict:
            if k in partial_params:
                completed_params[k] = partial_params[k]
            else:
                completed_params[k] = jnp.full(shape=(n_population,), fill_value=frozen_params_dict[k])
        return params.__class__(**completed_params)

    es = CMA_ES(population_size=n_population, solution=optimizable_params_dict)
    es_params = es.default_params
    key, subkey = random.split(key)
    es_state = es.init(subkey, optimizable_params_dict, es_params)

    metrics_logs = []
    for _ in tqdm(range(n_generations), desc="Optimizing...", total=n_generations, disable=not progbar):
        key, subkey = random.split(key)
        key_ask, key_eval, key_tell = random.split(subkey, 3)

        population, es_state = es.ask(key_ask, es_state, es_params)
        population_completed = complete_params(population)

        _, predictions = predict(population_completed, obs, step_fn)
        err = jax.tree.map(lambda p,t: p-t.reshape(-1,1), predictions, target)
        if pinball_alpha is not None:
            score_individual = jax.tree.map(
                lambda x: jnp.where(err > 0, err * (1-pinball_alpha), -err * pinball_alpha),
                err
            )
        else:
            score_individual = jax.tree.map(jnp.square, err)
        score_weighted = jax.tree.map(lambda e,w: e*w.reshape(-1,1), score_individual, weights)
        score = jax.tree.map(lambda x: jnp.mean(x, axis=0), score_weighted)
        score_merged = jnp.nansum(jnp.stack(jax.tree.leaves(score), axis=0), axis=0)

        bias_correction_score = abssum_weight * jnp.abs(
            jnp.nansum(jnp.stack(jax.tree.leaves(jax.tree.map(lambda x: jnp.cumsum(x, axis=0)/x.shape[0], err)), axis=0), axis=(0,1))
        )

        es_state, metrics = es.tell(key_tell, population, score_merged + bias_correction_score, es_state, es_params)

        metrics_logs.append(metrics)

    best_params = {k:v for k,v in frozen_params_dict.items()}
    best_params.update(
        jax.tree.unflatten(jax.tree.structure(optimizable_params_dict), metrics_logs[-1]["best_solution"])
    )
    for k in best_params:
        best_params[k] = jnp.ravel(best_params[k])[0]
    best_params = params.__class__(**best_params)

    return best_params, metrics_logs
