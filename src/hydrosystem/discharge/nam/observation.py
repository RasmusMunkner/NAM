from jax import numpy as jnp
from typing import NamedTuple, Any
from jaxtyping import Float

class NAMObservation(NamedTuple):
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
    p: Float[Any, "dim"]  # Precipitation
    epot: Float[Any, "dim"]  # Potential Evapotranspiration
    t: Float[Any, "dim"]  # Temperature (Celsius)

    def canonical(self):
        p = jnp.ravel(jnp.asarray(self.p))
        epot = jnp.ravel(jnp.asarray(self.epot))
        t = jnp.ravel(jnp.asarray(self.t))
        if p.shape[0] != epot.shape[0] or p.shape[0] != t.shape[0]:
            raise ValueError('NAMObservation fields must have identical shapes.')
        return NAMObservation(p, epot, t)


class NAMTarget(NamedTuple):
    """Class for holding output data to hydrological models."""
    q: Float[Any, "dim"]
    eact: Float[Any, "dim"]
    perc: Float[Any, "dim"]
    recharge: Float[Any, "dim"]
    storage: Float[Any, "dim"]

    @classmethod
    def from_partial(
            cls,
            q: Float[Any, "dim"],
            eact: Float[Any, "dim"] = None,
            perc: Float[Any, "dim"] = None,
            storage: Float[Any, "dim"] = None,
    ):
        """Build a hydrological target vector from partial outputs."""
        for x in [q, eact, perc, storage]:
            if x is not None:
                if x.shape != q.shape:
                    raise ValueError('NAMTarget fields must have identical shapes.')

        return cls(
            q = jnp.asarray(q),
            eact = jnp.asarray(eact) if eact is not None else jnp.full(q.shape, jnp.nan),
            perc = jnp.asarray(perc) if perc is not None else jnp.full(q.shape, jnp.nan),
            storage = jnp.asarray(storage) if storage is not None else jnp.full(q.shape, jnp.nan),
        )














