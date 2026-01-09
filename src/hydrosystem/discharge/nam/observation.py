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

    @classmethod
    def from_partial(cls, q: Float[Any, "dim"] = None, eact: Float[Any, "dim"] = None, perc: Float[Any, "dim"] = None):
        """Build a hydrological target vector from partial outputs."""
        if q is None and eact is None and perc is None:
            raise ValueError("Either q or eact or perc must be given.")
        if q is not None:
            target_shape = q.shape
        elif eact is not None:
            target_shape = eact.shape
        else:
            target_shape = perc.shape

        for x in [q, eact, perc]:
            if x is not None:
                if x.shape != target_shape:
                    raise ValueError('NAMTarget fields must have identical shapes.')

        return cls(
            q = jnp.asarray(q) if q is not None else jnp.full(target_shape, jnp.nan),
            eact = jnp.asarray(eact) if eact is not None else jnp.full(target_shape, jnp.nan),
            perc = jnp.asarray(perc) if perc is not None else jnp.full(target_shape, jnp.nan)
        )














