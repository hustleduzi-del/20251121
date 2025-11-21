"""
Monte Carlo simulation for pricing options with custom payoff functions.

The module provides a small, dependency-free implementation that can price
options under a geometric Brownian motion model. Users supply a payoff
function that consumes the simulated path, enabling payoffs that depend on
terminal or path-dependent values.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class MonteCarloSpec:
    """Inputs required to run a Monte Carlo option pricing simulation."""

    spot: float
    maturity: float
    rate: float
    volatility: float
    simulations: int = 10_000
    steps: int = 1
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("Spot must be positive")
        if self.maturity <= 0:
            raise ValueError("Maturity must be positive")
        if self.volatility <= 0:
            raise ValueError("Volatility must be positive")
        if self.simulations <= 0:
            raise ValueError("Number of simulations must be positive")
        if self.steps <= 0:
            raise ValueError("Number of time steps must be positive")


def monte_carlo_price(spec: MonteCarloSpec, payoff_fn: Callable[[Sequence[float]], float]) -> float:
    """Price an option via Monte Carlo simulation with a custom payoff.

    The simulated asset follows a geometric Brownian motion discretized into
    ``steps`` equal intervals. The payoff function receives the full simulated
    path, so users can express vanilla or path-dependent payoffs.

    Args:
        spec: Simulation parameters.
        payoff_fn: Callable that accepts the simulated price path (including
            the initial price) and returns the payoff for that path.

    Returns:
        Present value of the expected payoff under the risk-neutral measure.
    """

    rng = random.Random(spec.seed)
    dt = spec.maturity / spec.steps
    drift = (spec.rate - 0.5 * spec.volatility ** 2) * dt
    diffusion = spec.volatility * math.sqrt(dt)

    total_payoff = 0.0
    for _ in range(spec.simulations):
        price = spec.spot
        path = [price]
        for _ in range(spec.steps):
            z = rng.gauss(0.0, 1.0)
            price *= math.exp(drift + diffusion * z)
            path.append(price)
        total_payoff += payoff_fn(path)

    expected_payoff = total_payoff / spec.simulations
    discount_factor = math.exp(-spec.rate * spec.maturity)
    return discount_factor * expected_payoff


def european_call_payoff(strike: float) -> Callable[[Sequence[float]], float]:
    """Factory for a European call payoff using the terminal price."""

    if strike <= 0:
        raise ValueError("Strike must be positive")

    def _payoff(path: Sequence[float]) -> float:
        terminal_price = path[-1]
        return max(terminal_price - strike, 0.0)

    return _payoff


def european_put_payoff(strike: float) -> Callable[[Sequence[float]], float]:
    """Factory for a European put payoff using the terminal price."""

    if strike <= 0:
        raise ValueError("Strike must be positive")

    def _payoff(path: Sequence[float]) -> float:
        terminal_price = path[-1]
        return max(strike - terminal_price, 0.0)

    return _payoff


def example() -> None:
    spec = MonteCarloSpec(
        spot=100,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        simulations=20_000,
        steps=1,
        seed=123,
    )

    call_price = monte_carlo_price(spec, european_call_payoff(strike=100))
    put_price = monte_carlo_price(spec, european_put_payoff(strike=100))

    # Example of a custom payoff: an average-price Asian call
    def asian_call(path: Sequence[float]) -> float:
        average_price = sum(path) / len(path)
        return max(average_price - 100, 0.0)

    asian_price = monte_carlo_price(spec, asian_call)

    print(f"European call (MC): {call_price:.4f}")
    print(f"European put  (MC): {put_price:.4f}")
    print(f"Asian call     (MC): {asian_price:.4f}")


if __name__ == "__main__":
    example()
