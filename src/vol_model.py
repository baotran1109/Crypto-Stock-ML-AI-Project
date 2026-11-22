"""
vol_model.py
Big-move (volatility) model, one per asset class. This is the prediction the research found
to work: "will the next 5 days move more than usual, in either direction?"

- Label: |5-day return| > the symbol's trailing 252-day median of |5-day returns|.
  By construction about half of all days are "big move" days, so 50% is the base rate.
- Features: the research feature set that scored best and needs no cross-sectional ranks,
  so a single symbol can be scored live (crypto: +sentiment, stocks: +events).
- Scoring: walk-forward by date (same folds as src/research.py) plus a final hold-out
  of the last config.model.test_fraction of dates. The saved model is then refit on all
  labelled data for live use.

Run from the project root:
    python -m src.vol_model                  # both asset classes
    python -m src.vol_model --classes stock
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src import research
from src.config import MODEL
from src.model_registry import TRAINING_UNIVERSES, artifact_dir, artifact_path

TARGET = "y_big5"
HORIZON = 5
FEATURE_SET = {"crypto": "+sentiment", "stock": "+events"}
LEVEL_NAMES = ["Low", "Below normal", "Normal", "Above normal", "High"]


def vol_features(asset_class: str):
    return research.feature_sets(asset_class)[FEATURE_SET[asset_class]]


def build_model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                          min_samples_leaf=200, l2_regularization=1.0, random_state=0)


def train_vol(asset_class: str) -> Dict:
    print(f"\n# {asset_class.upper()} BIG-MOVE MODEL")
    context = research.load_context()
    panel = research.build_panel(asset_class, context)
    feats = vol_features(asset_class)
    data = panel.dropna(subset=[TARGET]).reset_index(drop=True)

    wf = research.walk_forward(panel, "bigmove5", feats, "hgb")
    print(f"   walk-forward ROC-AUC {wf['mean_auc']:.3f} (worst {wf['min_auc']:.3f}, "
          f"{wf['folds_above_50']}/{wf['n_folds']} periods > 0.5)")

    dates = np.sort(data["date"].unique())
    cut = int(len(dates) * (1 - MODEL["test_fraction"]))
    test_start = pd.Timestamp(dates[cut])
    train = data[data["date"] < pd.Timestamp(dates[max(cut - HORIZON - 1, 0)])]
    test = data[data["date"] >= test_start]
    m = build_model().fit(train[feats], train[TARGET].astype(int))
    p = m.predict_proba(test[feats])[:, 1]
    y = test[TARGET].astype(int).values
    auc = roc_auc_score(y, p)

    # calibration: when the model says X%, how often was there really a big move?
    bins = pd.qcut(p, 5, labels=False, duplicates="drop")
    calib = (pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin")
             .agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size")).reset_index(drop=True))
    print(f"   hold-out ROC-AUC {auc:.3f} ({test_start.date()} -> {pd.Timestamp(dates[-1]).date()})")
    print(calib.round(3).to_string(index=False))

    final = build_model().fit(data[feats], data[TARGET].astype(int))
    # The app reports levels, not raw probabilities (raw scores were not well calibrated).
    # Level = which fifth of recent scores today's score falls in; each level's hit rate is
    # the real big-move frequency in that fifth during the hold-out test.
    recent = data[data["date"] >= pd.Timestamp(dates[-1]) - pd.Timedelta(days=365)]
    score_edges = np.quantile(final.predict_proba(recent[feats])[:, 1], [0.2, 0.4, 0.6, 0.8]).tolist()
    metrics = {
        "score_edges": score_edges,
        "level_hit_rate": calib["actual"].tolist(),
        "holdout_roc_auc": float(auc),
        "walk_forward": {k: wf[k] for k in ["mean_auc", "min_auc", "folds_above_50", "n_folds", "folds"]},
        "base_rate": float(data[TARGET].mean()),
        "calibration": calib.to_dict("records"),
        "test_start": str(test_start.date()),
        "test_end": str(pd.Timestamp(dates[-1]).date()),
    }
    meta = {
        "asset_class": asset_class,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "symbols": sorted(data["symbol"].unique().tolist()),
        "feature_set": FEATURE_SET[asset_class],
        "feature_cols": feats,
        "data_start": str(pd.Timestamp(dates[0]).date()),
        "data_end": str(pd.Timestamp(dates[-1]).date()),
        "note": "Final model is refit on all labelled data; scores above come from earlier fits that never saw the test dates.",
    }
    os.makedirs(artifact_dir(asset_class), exist_ok=True)
    joblib.dump(final, artifact_path(asset_class, "vol_model.pkl"))
    joblib.dump(metrics, artifact_path(asset_class, "vol_metrics.pkl"))
    joblib.dump(meta, artifact_path(asset_class, "vol_metadata.pkl"))
    print(f"   [SAVE] {artifact_dir(asset_class)}/vol_model.pkl")
    return metrics


def symbol_frame(symbol: str, asset_class: str, fresh_hours: float = MODEL["fresh_hours"]) -> pd.DataFrame:
    """All research features for one symbol, ready for vol_model.predict_proba."""
    f = research.asset_frame(symbol, fresh_hours=fresh_hours)
    if f.empty:
        return f
    context = research.load_context(fresh_hours=fresh_hours)
    return research.attach_shared(f, asset_class, context).replace([np.inf, -np.inf], np.nan)


def predict_big_move(symbol: str, asset_class: str, bundle: Dict) -> Optional[Dict]:
    """Latest P(big move in the next 5 bars) for one symbol, or None if it can't be computed."""
    model, feats = bundle.get("vol_model"), (bundle.get("vol_metadata") or {}).get("feature_cols")
    if model is None or not feats:
        return None
    f = symbol_frame(symbol, asset_class)
    if f.empty:
        return None
    f = f.dropna(subset=["rsi", "vol20", "move_thr"])
    if f.empty:
        return None
    last = f.iloc[[-1]]
    prob = float(model.predict_proba(last[feats])[:, 1][0])
    hist = model.predict_proba(f[feats].tail(250))[:, 1]
    metrics = bundle.get("vol_metrics") or {}
    edges = metrics.get("score_edges")
    level = int(np.searchsorted(edges, prob)) if edges else None   # 0..4
    hit = metrics.get("level_hit_rate")
    return {
        "prob": prob,
        "level": level,
        "level_name": LEVEL_NAMES[level] if level is not None else None,
        "hit_rate": float(hit[level]) if hit and level is not None and level < len(hit) else None,
        "base_rate": metrics.get("base_rate"),
        "date": pd.Timestamp(last["date"].iloc[0]),
        "typical_move": float(last["move_thr"].iloc[0]),
        "pct_rank_1y": float((hist < prob).mean()),
        "history": pd.Series(hist, index=pd.DatetimeIndex(f["date"].tail(250))),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=list(TRAINING_UNIVERSES), choices=list(TRAINING_UNIVERSES))
    args = ap.parse_args()
    for ac in args.classes:
        train_vol(ac)
