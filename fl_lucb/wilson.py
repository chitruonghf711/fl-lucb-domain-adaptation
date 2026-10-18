"""Wilson confidence intervals used by the LUCB selector."""

from __future__ import annotations

from statistics import NormalDist

from .models import WilsonInterval


def wilson_interval(successes: int, total: int, delta: float = 0.05) -> WilsonInterval:
    """Return a two-sided Wilson interval for a Bernoulli success rate.

    ``delta`` is the total tail probability, so a default of 0.05 produces a
    95 percent interval. Empty evidence remains explicitly uninformative.
    """

    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes and total must satisfy 0 <= successes <= total")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if total == 0:
        return WilsonInterval(
            successes=successes,
            total=total,
            estimate=0.0,
            lower=0.0,
            upper=1.0,
            delta=delta,
        )

    estimate = successes / total
    z = NormalDist().inv_cdf(1 - delta / 2)
    z_squared = z * z
    denominator = 1 + z_squared / total
    centre = (estimate + z_squared / (2 * total)) / denominator
    radius = (
        z
        * ((estimate * (1 - estimate) / total) + z_squared / (4 * total * total)) ** 0.5
        / denominator
    )
    return WilsonInterval(
        successes=successes,
        total=total,
        estimate=estimate,
        lower=max(0.0, centre - radius),
        upper=min(1.0, centre + radius),
        delta=delta,
    )


def error_interval(errors: int, total: int, delta: float = 0.05) -> WilsonInterval:
    """Return a Wilson interval for an error rate instead of accuracy."""

    return wilson_interval(successes=errors, total=total, delta=delta)
