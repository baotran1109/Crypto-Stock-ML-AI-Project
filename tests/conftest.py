import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def make_ohlcv(n: int = 600, seed: int = 0, start: str = "2020-01-01", freq: str = "D", scale: float = 100.0):
    """Synthetic random-walk OHLCV bars."""
    rng = np.random.default_rng(seed)
    close = scale * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    vol = rng.uniform(1e5, 1e6, n)
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


@pytest.fixture
def ohlcv():
    return make_ohlcv()
