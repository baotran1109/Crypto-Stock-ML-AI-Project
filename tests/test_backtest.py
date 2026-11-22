import numpy as np
import pytest

pytest.importorskip("backtrader")

from src.backtest import _max_drawdown, _run_engine  # noqa: E402
from tests.conftest import make_ohlcv  # noqa: E402


def _risk(**kw):
    r = {"commission": 0.0, "slippage": 0.0, "vol_target": None, "min_hold": 0, "periods_per_year": 365}
    r.update(kw)
    return r


def test_buy_and_hold_matches_price_return():
    df = make_ohlcv(300)
    strat = _run_engine(df, np.ones(len(df), dtype=int), 10000, _risk())
    expected = df["close"].iloc[-1] / df["close"].iloc[0]
    # sizing keeps a 0.1% cash buffer
    assert strat.final_value / 10000 == pytest.approx(expected, rel=0.01)


def test_flat_signal_never_trades():
    df = make_ohlcv(200)
    strat = _run_engine(df, np.zeros(len(df), dtype=int), 10000, _risk(commission=0.001))
    assert strat.final_value == pytest.approx(10000)
    assert strat.trades_list == []


def test_costs_reduce_returns():
    df = make_ohlcv(300)
    sig = np.tile([1, 1, 0, 0], len(df) // 4)
    free = _run_engine(df, sig, 10000, _risk()).final_value
    costly = _run_engine(df, sig, 10000, _risk(commission=0.002)).final_value
    assert costly < free


def test_vol_target_scales_position_down():
    df = make_ohlcv(300)
    vols = np.full(len(df), 0.05)  # 5% daily, about 95% annualised
    strat = _run_engine(df, np.ones(len(df), dtype=int), 10000, _risk(vol_target=0.2, vols=vols))
    assert strat.trades_list[0]["position_fraction"] == pytest.approx(0.2 / (0.05 * np.sqrt(365)), rel=1e-6)


def test_min_hold_delays_exit():
    df = make_ohlcv(100)
    sig = np.full(len(df), -1)
    sig[10], sig[11] = 1, 0
    strat = _run_engine(df, sig, 10000, _risk(min_hold=5))
    # exit signal on bar 11 is ignored (held 1 bar < 5), so the trade stays open to the end
    assert strat.trades_list[0]["open_at_end"]


def test_max_drawdown():
    assert _max_drawdown(np.array([100, 120, 90, 130])) == pytest.approx(0.25)
