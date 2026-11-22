import pandas as pd

from src.features import FEATURE_COLS, HORIZON, add_indicators, make_supervised
from tests.conftest import make_ohlcv


def test_no_lookahead_in_indicators(ohlcv):
    """A feature on date d must not change when later bars are added."""
    full = add_indicators(ohlcv)[FEATURE_COLS]
    part = add_indicators(ohlcv.iloc[:400])[FEATURE_COLS]
    pd.testing.assert_frame_equal(full.loc[part.index], part, check_exact=False, rtol=1e-9)


def test_features_are_scale_free():
    """Same price path at $1 and $60,000 gives the same model features."""
    a = add_indicators(make_ohlcv(scale=1.0))[FEATURE_COLS]
    b = add_indicators(make_ohlcv(scale=60000.0))[FEATURE_COLS]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-6)


def test_make_supervised_label_and_tail(ohlcv):
    df = add_indicators(ohlcv)
    X, y = make_supervised(df)
    # the last HORIZON bars have no known outcome and must be dropped
    assert X.index.max() <= df.index[-HORIZON - 1]
    future = df["close"].shift(-HORIZON)
    expected = (future.loc[X.index] > df["close"].loc[X.index]).astype(int)
    assert (y.values == expected.values).all()
    assert list(X.columns) == FEATURE_COLS
