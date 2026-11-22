import numpy as np
import pandas as pd

from src import alt_data, research
from tests.conftest import make_ohlcv

LABELS = {"fwd5", "fwd10", "fwd20", "absfwd5", "y_big5", "y_dir5", "y_dir20", "symbol"}


def _patch(monkeypatch, df):
    monkeypatch.setattr(research, "_load", lambda sym, fresh_hours=0: df)
    monkeypatch.setattr(alt_data, "stock_event_features", lambda s, d: pd.DataFrame(index=d))
    monkeypatch.setattr(alt_data, "coin_funding_features", lambda c, d: pd.DataFrame(index=d))


def test_asset_frame_features_have_no_lookahead(monkeypatch):
    df = make_ohlcv(700)
    _patch(monkeypatch, df)
    full = research.asset_frame("XYZ").set_index("date")
    _patch(monkeypatch, df.iloc[:500])
    part = research.asset_frame("XYZ").set_index("date")
    feats = [c for c in part.columns if c not in LABELS]
    pd.testing.assert_frame_equal(full.loc[part.index, feats], part[feats], check_exact=False, rtol=1e-9)


def test_labels_unknown_at_the_end(monkeypatch):
    _patch(monkeypatch, make_ohlcv(400))
    f = research.asset_frame("XYZ")
    assert f["y_dir5"].tail(5).isna().all()
    assert f["y_dir20"].tail(20).isna().all()


def _panel(n_dates=800, n_sym=5, seed=0, leak=False):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_dates)
    rows = []
    for s in range(n_sym):
        y = rng.integers(0, 2, n_dates).astype(float)
        x = y + rng.normal(0, 0.5, n_dates) if leak else rng.normal(size=n_dates)
        rows.append(pd.DataFrame({"date": dates, "symbol": f"S{s}", "x": x, "y_dir5": y,
                                  "fwd5": rng.normal(size=n_dates), "ret20": rng.normal(size=n_dates),
                                  "vol20": rng.random(n_dates)}))
    return pd.concat(rows, ignore_index=True)


def test_walk_forward_random_feature_scores_near_half():
    res = research.walk_forward(_panel(), "dir5", ["x"], "logreg")
    assert res["n_folds"] == research.N_FOLDS
    assert abs(res["mean_auc"] - 0.5) < 0.05


def test_walk_forward_detects_real_signal():
    res = research.walk_forward(_panel(leak=True), "dir5", ["x"], "logreg")
    assert res["mean_auc"] > 0.8
