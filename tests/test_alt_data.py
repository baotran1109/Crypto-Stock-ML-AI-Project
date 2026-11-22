import numpy as np
import pandas as pd

from src import alt_data


def test_asof_uses_last_known_value():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2024-01-05"]))
    dates = pd.date_range("2023-12-31", "2024-01-06")
    out = alt_data._asof(s, dates)
    assert np.isnan(out.iloc[0])
    assert out.loc["2024-01-04"] == 1.0
    assert out.loc["2024-01-05"] == 2.0


def _fake_earnings(rows):
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in rows], name="date")
    return pd.DataFrame({"eps_est": 1.0, "eps": 1.0, "surprise_pct": [v for _, v in rows]}, index=idx)


def test_earnings_known_only_from_next_day(monkeypatch):
    # report after the US close (21:00 UTC) on 2024-02-01
    monkeypatch.setattr(alt_data, "earnings",
                        lambda s: _fake_earnings([("2023-11-01 21:00", 5.0), ("2024-02-01 21:00", 20.0)]))
    monkeypatch.setattr(alt_data, "analyst_actions", lambda s: pd.DataFrame())
    dates = pd.bdate_range("2024-01-29", "2024-02-06")
    f = alt_data.stock_event_features("XYZ", dates)
    assert f.loc["2024-02-01", "eps_surprise"] == 5.0       # not yet public at that day's close
    assert f.loc["2024-02-02", "eps_surprise"] == 20.0
    assert f.loc["2024-02-02", "eps_surprise_prev"] == 5.0
    assert f.loc["2024-02-02", "days_since_earn"] == 0


def test_event_features_ignore_future_events(monkeypatch):
    base = [("2023-05-01 21:00", 3.0), ("2023-08-01 21:00", -2.0)]
    dates = pd.bdate_range("2023-04-01", "2023-09-30")
    monkeypatch.setattr(alt_data, "analyst_actions", lambda s: pd.DataFrame())
    monkeypatch.setattr(alt_data, "earnings", lambda s: _fake_earnings(base))
    a = alt_data.stock_event_features("XYZ", dates)
    monkeypatch.setattr(alt_data, "earnings", lambda s: _fake_earnings(base + [("2023-11-01 21:00", 50.0)]))
    b = alt_data.stock_event_features("XYZ", dates)
    pd.testing.assert_frame_equal(a, b)


def test_analyst_actions_known_only_from_next_day(monkeypatch):
    acts = pd.DataFrame({"Action": ["up", "down"], "priceTargetAction": ["Raises", ""]},
                        index=pd.DatetimeIndex(["2023-01-02 13:00", "2024-03-05 13:00"], name="date"))
    monkeypatch.setattr(alt_data, "earnings", lambda s: pd.DataFrame())
    monkeypatch.setattr(alt_data, "analyst_actions", lambda s: acts)
    f = alt_data.stock_event_features("XYZ", pd.bdate_range("2024-03-01", "2024-03-08"))
    assert f.loc["2024-03-05", "analyst_net30"] == 0
    assert f.loc["2024-03-06", "analyst_net30"] == -1


def test_funding_features_no_lookahead(monkeypatch):
    rng = np.random.default_rng(1)
    days = pd.date_range("2023-06-01", periods=300)
    full = pd.DataFrame({"funding": rng.normal(1e-4, 5e-5, len(days))}, index=days)
    monkeypatch.setattr(alt_data, "hyperliquid_funding", lambda c: full)
    a = alt_data.coin_funding_features("ABC", days)
    monkeypatch.setattr(alt_data, "hyperliquid_funding", lambda c: full.iloc[:200])
    b = alt_data.coin_funding_features("ABC", days[:200])
    pd.testing.assert_frame_equal(a.iloc[:200], b, check_freq=False)
