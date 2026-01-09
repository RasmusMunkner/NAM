
from jax import numpy as jnp
from jaxtyping import Float
from typing import NamedTuple, Any
import pymannkendall as mk
from matplotlib import pyplot as plt

class MannKendallAnalysisResults(NamedTuple):
    x: Float[Any, "dim"]
    trend: str
    p: Float[Any, ""]
    intercept: Float[Any, ""]
    slope: Float[Any, ""]

    def plot(self, t: Float[Any, "dim_t"] = None, ax: plt.Axes = None) -> None:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        t_ = jnp.arange(self.x.shape[0])
        if t is None:
            t = t_
        t = jnp.asarray(t)
        ax.scatter(t, self.x, s=5)
        x_trendline = jnp.array([jnp.min(t), jnp.max(t)])
        y_trendline = self.intercept + self.slope * jnp.array([jnp.min(t_), jnp.max(t_)])
        ax.plot(x_trendline, y_trendline, linestyle="--", color="red", label=f"p={self.p:.2e}, slope={self.slope:.2e}")

        if fig is not None:
            fig.show()

    def significant(self, alpha: float) -> bool:
        return self.p < alpha

        
def mann_kendall_analysis(x: Float[Any, "dim"]) -> MannKendallAnalysisResults:
    trend,h,p,z,tau,s,var_s,slope,intercept = mk.original_test(x)
    return MannKendallAnalysisResults(
        x, trend, p, intercept, slope
    )


