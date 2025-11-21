"""
Simple implementation of the Cox-Ross-Rubinstein binomial option pricing model.

Supports European and American call/put options.  The implementation exposes a
convenient dataclass for capturing option parameters and a pricing function that
performs backward induction on the binomial tree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

OptionType = Literal["call", "put"]
ExerciseType = Literal["european", "american"]


@dataclass(frozen=True)
class OptionSpec:
    """Container for the inputs required by the binomial model."""

    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    steps: int
    option_type: OptionType = "call"
    exercise: ExerciseType = "european"

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("Number of steps must be positive")
        if self.maturity <= 0:
            raise ValueError("Maturity must be positive")
        if self.volatility <= 0:
            raise ValueError("Volatility must be positive")
        if self.strike <= 0:
            raise ValueError("Strike must be positive")
        if self.spot <= 0:
            raise ValueError("Spot must be positive")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        if self.exercise not in ("european", "american"):
            raise ValueError("exercise must be 'european' or 'american'")


def payoff(price: float, strike: float, option_type: OptionType) -> float:
    if option_type == "call":
        return max(price - strike, 0.0)
    return max(strike - price, 0.0)


def binomial_price(spec: OptionSpec) -> float:
    """Price an option using the CRR binomial model.

    Args:
        spec: Option specification containing market parameters and contract
            details.

    Returns:
        The present value of the option.
    """

    dt = spec.maturity / spec.steps
    u = math.exp(spec.volatility * math.sqrt(dt))
    d = 1 / u
    disc = math.exp(-spec.rate * dt)
    p = (math.exp(spec.rate * dt) - d) / (u - d)

    if not 0.0 < p < 1.0:
        raise ValueError("Risk-neutral probability is outside (0, 1). Check inputs.")

    # Terminal node payoffs
    values = [
        payoff(spec.spot * (u ** j) * (d ** (spec.steps - j)), spec.strike, spec.option_type)
        for j in range(spec.steps + 1)
    ]

    # Roll back through the tree
    for step in range(spec.steps - 1, -1, -1):
        next_values = []
        for up_moves in range(step + 1):
            continuation = disc * (p * values[up_moves + 1] + (1 - p) * values[up_moves])
            if spec.exercise == "american":
                price_at_node = spec.spot * (u ** up_moves) * (d ** (step - up_moves))
                exercise_value = payoff(price_at_node, spec.strike, spec.option_type)
                next_values.append(max(continuation, exercise_value))
            else:
                next_values.append(continuation)
        values = next_values

    return values[0]


def example() -> None:
    spec = OptionSpec(
        spot=100,
        strike=100,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=3,
        option_type="call",
        exercise="european",
    )
    price = binomial_price(spec)
    print(f"European call price (3-step tree): {price:.4f}")


if __name__ == "__main__":
    example()
