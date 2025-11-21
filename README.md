# 20251121

Python implementation of the Cox–Ross–Rubinstein binomial option pricing model. It
supports both European and American call/put options via a simple API.

## Usage

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
