"""
model_registry.py
One model per asset class. Crypto and stocks behave differently (24/7 vs weekday trading,
3-5x different volatility, different drivers), so each gets a model trained on its own data.

Artifacts live in artifacts/<asset_class>/:
    model.pkl, feature_columns.pkl, evaluation_metrics.pkl, feature_importance.pkl, training_metadata.pkl
    vol_model.pkl, vol_metrics.pkl, vol_metadata.pkl   (big-move model, src/vol_model.py)
"""

import os
from typing import Dict, Optional

import joblib

from src.config import UNIVERSES

ARTIFACTS_ROOT = "artifacts"

# Symbols each model is trained on
TRAINING_UNIVERSES = {ac: list(syms) for ac, syms in UNIVERSES.items()}  # from config.toml

ASSET_CLASS_LABELS = {"crypto": "Crypto", "stock": "Stocks"}


def asset_class_for(symbol: str) -> str:
    """Yahoo crypto tickers end in -USD (BTC-USD); everything else is treated as a stock/ETF."""
    return "crypto" if symbol.strip().upper().endswith("-USD") else "stock"


def artifact_dir(asset_class: str) -> str:
    return os.path.join(ARTIFACTS_ROOT, asset_class)


def artifact_path(asset_class: str, name: str) -> str:
    return os.path.join(artifact_dir(asset_class), name)


def load_artifacts(asset_class: str) -> Optional[Dict]:
    """Return the model bundle for an asset class, or None if it hasn't been trained."""
    try:
        bundle = {
            "model": joblib.load(artifact_path(asset_class, "model.pkl")),
            "feature_cols": joblib.load(artifact_path(asset_class, "feature_columns.pkl")),
        }
    except (FileNotFoundError, OSError):
        return None
    for key, fname in [("metrics", "evaluation_metrics.pkl"),
                       ("importance", "feature_importance.pkl"),
                       ("metadata", "training_metadata.pkl"),
                       # big-move (volatility) model from src/vol_model.py, optional
                       ("vol_model", "vol_model.pkl"),
                       ("vol_metrics", "vol_metrics.pkl"),
                       ("vol_metadata", "vol_metadata.pkl")]:
        try:
            bundle[key] = joblib.load(artifact_path(asset_class, fname))
        except (FileNotFoundError, OSError):
            bundle[key] = None
    return bundle
