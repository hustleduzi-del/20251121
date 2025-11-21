from __future__ import annotations

from typing import Any, Dict

from flask import Flask, render_template, request, url_for

from monte_carlo import MonteCarloSpec, price_european_option

app = Flask(__name__)


def _parse_float(value: str, name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _parse_int(value: str, name: str) -> int:
    try:
        number = int(float(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return number


def _build_spec(form: Dict[str, str]) -> MonteCarloSpec:
    spot = _parse_float(form.get("spot", ""), "Spot")
    maturity = _parse_float(form.get("maturity", ""), "Maturity")
    rate = _parse_float(form.get("rate", ""), "Rate")
    volatility = _parse_float(form.get("volatility", ""), "Volatility")
    simulations = _parse_int(form.get("simulations", ""), "Simulations")
    steps = _parse_int(form.get("steps", ""), "Steps")
    seed_value = form.get("seed")
    seed: int | None
    if seed_value:
        seed = _parse_int(seed_value, "Seed")
    else:
        seed = None

    return MonteCarloSpec(
        spot=spot,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        simulations=simulations,
        steps=steps,
        seed=seed,
    )


@app.route("/", methods=["GET"])
def index() -> str:
    default_values: Dict[str, Any] = {
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "rate": 0.05,
        "volatility": 0.2,
        "simulations": 20000,
        "steps": 1,
        "option_type": "call",
        "seed": "",
    }
    return render_template("index.html", defaults=default_values, result=None, error=None)


@app.route("/price", methods=["POST"])
def price() -> str:
    form = request.form
    try:
        option_type = form.get("option_type", "call").lower()
        strike = _parse_float(form.get("strike", ""), "Strike")
        spec = _build_spec(form) 
        price_value = price_european_option(spec, strike=strike, option_type=option_type)
    except Exception as exc:  # noqa: BLE001
        default_values = {**form}
        return render_template("index.html", defaults=default_values, result=None, error=str(exc)), 400

    result = {
        "price": price_value,
        "spot": spec.spot,
        "strike": strike,
        "maturity": spec.maturity,
        "rate": spec.rate,
        "volatility": spec.volatility,
        "steps": spec.steps,
        "simulations": spec.simulations,
        "option_type": option_type,
        "seed": spec.seed,
    }

    return render_template("index.html", defaults=form, result=result, error=None)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
