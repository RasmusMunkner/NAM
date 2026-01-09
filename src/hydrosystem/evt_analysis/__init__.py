from jax import numpy as jnp
import scipy
from typing import NamedTuple, Any
from jaxtyping import Float
from matplotlib import pyplot as plt


class GumbelAnalysisResults(NamedTuple):
    t: Float[Any, "dim_t"]
    event: Float[Any, "dim_t"]
    std: Float[Any, "dim_t"]

    def plot(self, ax: tuple[plt.Axes, plt.Axes] = None, low: float = 0.025, high: float = 0.975) -> None:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        qq_dd = gaussian_qq_dd(self.event, self.std, low=low, high=high)

        for i, (t, e) in enumerate(zip(self.t, self.event)):
            (line,) = ax.plot(qq_dd[i, :, 0], qq_dd[i, :, 1], label=f"{int(round(100*(high-low)))}%-CI for {t}-year-event")
            ax.axvline(x=e, linestyle="--", color=line.get_color(), label=f"{t}-year-event")


        if fig is not None:
            fig.show()



def gumbel_analysis(x: Float[Any, "dim_x"], t: Float[Any, "dim_t"], minima: bool=False) -> GumbelAnalysisResults:
    """Fit a Gumbel distribution using method of moments."""
    x, t = jnp.asarray(x), jnp.asarray(t)

    if minima:
        x = -x

    # Find the first two moments
    mu = jnp.mean(x)
    sigma = jnp.std(x)

    t_year_event = mu + 0.78 * sigma * (-jnp.log(-jnp.log(1 - 1 / t)) - 0.577)
    if minima:
        t_year_event = -t_year_event
    # t_year_event_min = mu + 0.78 * sigma * (jnp.log(-jnp.log(1 - 1 / t)) + 0.577)

    kt = 0.78 * (-jnp.log(-jnp.log(1 - 1 / t)) - 0.577)
    # kt_min = 0.78 * (jnp.log(-jnp.log(1 - 1 / t)) + 0.577)

    st = sigma / jnp.sqrt(len(x)) * jnp.sqrt(1 + 1.14 * kt + 1.1 * kt ** 2)
    # st_min = sigma / jnp.sqrt(len(x)) * jnp.sqrt(1 - 1.14 * kt_min + 1.1 * kt_min ** 2)

    return GumbelAnalysisResults(t, t_year_event, st)


def gaussian_qq_dd(
        mu: Float[Any, "dim_t"],
        sigma: Float[Any, "dim_t"],
        low: float = 0.025,
        high: float = 0.975,
        n: int = 500
) -> Float[Any, "dim_t n 2"]:
    """Calculate Gaussian quantiles and corresponding density estimates at these."""
    qq = jnp.linspace(low, high, n).reshape(1,n)
    qgaus = scipy.stats.norm.ppf(qq, mu.reshape(-1,1), sigma.reshape(-1,1))
    dgaus = scipy.stats.norm.pdf(qgaus, mu.reshape(-1,1), sigma.reshape(-1,1))
    return jnp.stack([qgaus, dgaus], axis=-1)

