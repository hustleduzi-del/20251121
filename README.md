# 20251121

Python implementation of the Cox–Ross–Rubinstein binomial option pricing model. It
supports both European and American call/put options via a simple API.

## Binomial model usage

```python
from binomial_model import OptionSpec, binomial_price

spec = OptionSpec(
    spot=100,
    strike=100,
    maturity=1.0,
    rate=0.05,
    volatility=0.2,
    steps=100,
    option_type="call",
    exercise="american",
)

price = binomial_price(spec)
print(price)
```

You can also run the module directly to see a small example:

```bash
python binomial_model.py
```

## Monte Carlo pricing with custom payoffs

```python
from monte_carlo import (
    MonteCarloSpec,
    european_call_payoff,
    monte_carlo_price,
)

spec = MonteCarloSpec(
    spot=100,
    maturity=1.0,
    rate=0.05,
    volatility=0.2,
    simulations=50_000,
)

call_price = monte_carlo_price(spec, european_call_payoff(strike=100))
print(call_price)

# Define a custom payoff: fixed cash flow if the average price stays above 90

def average_above_90(path: list[float]) -> float:
    return 10.0 if sum(path) / len(path) > 90 else 0.0

custom_price = monte_carlo_price(spec, average_above_90)
print(custom_price)
```

Running the module directly prints Monte Carlo prices for example European and
Asian payoffs:

```bash
python monte_carlo.py
```

### Command-line pricing for European options

You can also pass option parameters on the command line. Defaults mirror the
examples above, so running without flags prices a one-year at-the-money call on
an asset priced at 100 with 20% volatility and a 5% risk-free rate:

```bash
python monte_carlo.py
```

Override any of the inputs to run custom scenarios:

```bash
python monte_carlo.py --spot 95 --strike 90 --maturity 0.5 \
  --rate 0.04 --volatility 0.25 --steps 12 --simulations 50000 --option-type put
```
