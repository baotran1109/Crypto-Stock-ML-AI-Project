"""
train_model.py
Trains one direction model PER ASSET CLASS (crypto, stock), each on its own symbols.

- Scale-free features (see src/features.py) so a model is not tied to one asset's price level
- Chronological train/test split by DATE across all symbols, with a purge gap of
  HORIZON bars so no training label looks into the test period
- Walk-forward cross-validation, also split by date
- Saves model, feature list, metrics, feature importance and training metadata
  (train/test date ranges) to artifacts/<asset_class>/ so the backtest can stay out-of-sample

Run from the project root:
    python -m src.train_model                 # both asset classes
    python -m src.train_model --classes stock
"""

import argparse
import os
from datetime import datetime
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from src.data_loader import load_crypto, _cache_path, _is_fresh
from src.features import FEATURE_COLS, HORIZON, add_indicators, make_supervised
from src.config import MODEL
from src.model_registry import TRAINING_UNIVERSES, artifact_dir, artifact_path

# ==========================================================
# CONFIG
# ==========================================================
LOOKBACK = MODEL["train_lookback"]       # settings live in config.toml
TEST_FRACTION = MODEL["test_fraction"]
CV_FOLDS = MODEL["cv_folds"]
USE_XGB = False        # True to use XGBoost instead of Random Forest (pip install xgboost)
FORCE_REFRESH = False  # True to ignore the cache and download fresh data
FRESH_HOURS = MODEL["fresh_hours"]  # cache older than this is re-downloaded (stale cache used if offline)


def build_model():
    if USE_XGB:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=800, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1,
        )
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


def date_split(dates: pd.Series, cutoff: pd.Timestamp, gap_days: int):
    """Train = strictly before (cutoff - gap); test = on/after cutoff."""
    train_mask = dates < cutoff - pd.Timedelta(days=gap_days)
    test_mask = dates >= cutoff
    return train_mask.values, test_mask.values


def collect(symbols: List[str]) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        print(f"\n[FETCH] {sym}")
        cache = _cache_path(sym, "yahoo", LOOKBACK)
        if FORCE_REFRESH:
            print("   Force refresh - downloading")
        elif _is_fresh(cache, FRESH_HOURS):
            print("   Using fresh cache")
        elif os.path.exists(cache):
            print(f"   Cache older than {FRESH_HOURS}h - downloading (falls back to cache if offline)")
        else:
            print("   No cache - downloading")

        try:
            df = load_crypto(sym, lookback=LOOKBACK, force_refresh=FORCE_REFRESH, fresh_hours=FRESH_HOURS)
        except Exception as e:
            print(f"[WARN] Skipping {sym}: {e}")
            continue
        if df.attrs.get("stale_cache"):
            print("   [WARN] Download failed - using stale cache")
        print(f"   {len(df)} rows from {df['source'].iloc[-1]} ({df.index.min().date()} -> {df.index.max().date()})")

        X, y = make_supervised(add_indicators(df))
        if len(X) < 200:
            print(f"[WARN] Skipping {sym}: only {len(X)} usable samples")
            continue
        part = X.copy()
        part["target"] = y
        part["symbol"] = sym
        part["date"] = part.index
        frames.append(part.reset_index(drop=True))
        print(f"[OK] {sym}: {len(part)} samples")

    if not frames:
        raise RuntimeError("[ERROR] No valid data to train on. Check your network or the data/ cache.")
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def train_asset_class(asset_class: str) -> dict:
    symbols = TRAINING_UNIVERSES[asset_class]
    print("\n" + "#" * 60 + f"\n# {asset_class.upper()} MODEL  ({', '.join(symbols)})\n" + "#" * 60)
    data = collect(symbols)
    print(f"\n[DATA] Combined: {len(data)} rows, {data['symbol'].nunique()} symbols, "
          f"{data['date'].min().date()} -> {data['date'].max().date()}")

    # ---------------- chronological split ----------------
    unique_dates = np.sort(data["date"].unique())
    test_cutoff = pd.Timestamp(unique_dates[int(len(unique_dates) * (1 - TEST_FRACTION))])
    # purge gap in calendar days; stocks need ~1.5x the bar count to cover weekends
    gap_days = HORIZON if asset_class == "crypto" else int(np.ceil(HORIZON * 1.5))
    train_mask, test_mask = date_split(data["date"], test_cutoff, gap_days)
    train, test = data[train_mask], data[test_mask]
    X_train, y_train = train[FEATURE_COLS], train["target"]
    X_test, y_test = test[FEATURE_COLS], test["target"]
    train_end = train["date"].max()

    print(f"[SPLIT] Train: {len(train)} rows ({train['date'].min().date()} -> {train_end.date()})")
    print(f"[SPLIT] Test:  {len(test)} rows ({test_cutoff.date()} -> {test['date'].max().date()})")
    print(f"[SPLIT] UP ratio  train={y_train.mean():.3f}  test={y_test.mean():.3f}")

    # ---------------- walk-forward CV ----------------
    print(f"\n[CV] {CV_FOLDS}-fold walk-forward CV by date...")
    train_dates = np.sort(train["date"].unique())
    fold_edges = np.linspace(0, len(train_dates), CV_FOLDS + 2).astype(int)[1:]
    cv_scores, cv_aucs = [], []
    for i in range(CV_FOLDS):
        fold_start = pd.Timestamp(train_dates[fold_edges[i]])
        fold_end = pd.Timestamp(train_dates[min(fold_edges[i + 1], len(train_dates) - 1)])
        tr = train["date"] < fold_start - pd.Timedelta(days=gap_days)
        te = (train["date"] >= fold_start) & (train["date"] < fold_end)
        if tr.sum() == 0 or te.sum() == 0:
            continue
        m = build_model()
        m.fit(X_train[tr], y_train[tr])
        p = m.predict_proba(X_train[te])[:, 1]
        cv_scores.append(accuracy_score(y_train[te], (p > 0.5).astype(int)))
        if y_train[te].nunique() > 1:
            cv_aucs.append(roc_auc_score(y_train[te], p))
        print(f"   fold {i + 1}: test {fold_start.date()} -> {fold_end.date()}  "
              f"acc={cv_scores[-1]:.3f}  auc={cv_aucs[-1] if cv_aucs else float('nan'):.3f}")
    print(f"[CV] Mean accuracy {np.mean(cv_scores):.3f}, mean ROC-AUC {np.mean(cv_aucs):.3f}")

    # ---------------- final model ----------------
    print("\n[TRAIN] Training final model...")
    model = build_model()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])

    print(f"\n[EVAL] {asset_class} OUT-OF-SAMPLE TEST SET")
    print(f"  Baseline (majority class): {baseline_accuracy:.3f}")
    print(f"  Accuracy:  {acc:.3f}  ({acc - baseline_accuracy:+.3f} vs baseline)")
    print(f"  Precision: {precision:.3f}   Recall: {recall:.3f}   F1: {f1:.3f}")
    print(f"  ROC-AUC:   {roc_auc:.3f}  (0.5 = random)")

    per_symbol = {}
    for sym, g in test.groupby("symbol"):
        p = probs[(test["symbol"] == sym).values]
        per_symbol[sym] = {
            "n": int(len(g)),
            "accuracy": float(accuracy_score(g["target"], (p > 0.5).astype(int))),
            "roc_auc": float(roc_auc_score(g["target"], p)) if g["target"].nunique() > 1 else None,
        }
        auc_s = per_symbol[sym]["roc_auc"]
        print(f"   {sym:10s} n={len(g):4d}  acc={per_symbol[sym]['accuracy']:.3f}  "
              f"auc={auc_s if auc_s is not None else float('nan'):.3f}")
    print("\n" + classification_report(y_test, preds, target_names=["DOWN", "UP"], zero_division=0))

    eval_metrics = {
        "accuracy": float(acc),
        "baseline_accuracy": float(baseline_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "cv_accuracy": float(np.mean(cv_scores)) if cv_scores else None,
        "cv_roc_auc": float(np.mean(cv_aucs)) if cv_aucs else None,
        "confusion_matrix": cm.tolist(),
        "test_size": int(len(y_test)),
        "positive_class_ratio": float(y_test.mean()),
        "per_symbol": per_symbol,
    }
    metadata = {
        "asset_class": asset_class,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "symbols": sorted(data["symbol"].unique().tolist()),
        "horizon": HORIZON,
        "purge_gap_days": gap_days,
        "feature_cols": FEATURE_COLS,
        "train_start": str(train["date"].min().date()),
        "train_end": str(train_end.date()),
        "test_start": str(test_cutoff.date()),
        "test_end": str(test["date"].max().date()),
    }

    # ---------------- save ----------------
    os.makedirs(artifact_dir(asset_class), exist_ok=True)
    joblib.dump(model, artifact_path(asset_class, "model.pkl"))
    joblib.dump(list(FEATURE_COLS), artifact_path(asset_class, "feature_columns.pkl"))
    joblib.dump(eval_metrics, artifact_path(asset_class, "evaluation_metrics.pkl"))
    joblib.dump(metadata, artifact_path(asset_class, "training_metadata.pkl"))
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        joblib.dump(imp.to_dict(), artifact_path(asset_class, "feature_importance.pkl"))
    print(f"[SAVE] Artifacts written to {artifact_dir(asset_class)}/")
    return eval_metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=list(TRAINING_UNIVERSES),
                    choices=list(TRAINING_UNIVERSES))
    args = ap.parse_args()
    summary = {ac: train_asset_class(ac) for ac in args.classes}
    print("\n" + "=" * 60)
    for ac, m in summary.items():
        print(f"[DONE] {ac:6s} test ROC-AUC {m['roc_auc']:.3f} | accuracy {m['accuracy']:.3f} "
              f"vs baseline {m['baseline_accuracy']:.3f} | CV ROC-AUC {m['cv_roc_auc']:.3f}")
