"""
train_model.py
Trains one general direction model on several crypto symbols.

- Scale-free features (see src/features.py) so the model is not tied to one asset's price level
- Chronological train/test split by DATE across all symbols, with a purge gap of
  HORIZON bars so no training label looks into the test period
- Walk-forward cross-validation, also split by date
- Saves model, feature list, metrics, feature importance and training metadata
  (train/test date ranges) to artifacts/ so the backtest can stay out-of-sample

Run from the project root:  python -m src.train_model
"""

import os
from datetime import datetime

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

# ==========================================================
# CONFIG
# ==========================================================
ARTIFACTS = "artifacts"
COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "MATIC-USD"]
LOOKBACK = "5y"
TEST_FRACTION = 0.2
CV_FOLDS = 5
USE_XGB = False        # True to use XGBoost instead of Random Forest (pip install xgboost)
FORCE_REFRESH = False  # True to ignore the cache and download fresh data
FRESH_HOURS = 12       # cache older than this is re-downloaded (stale cache is used if offline)

os.makedirs(ARTIFACTS, exist_ok=True)


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


# ==========================================================
# COLLECT & PREPARE DATA
# ==========================================================
frames = []
for sym in COINS:
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

    feat = add_indicators(df)
    X, y = make_supervised(feat)
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

data = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
print(f"\n[DATA] Combined: {len(data)} rows, {data['symbol'].nunique()} symbols, "
      f"{data['date'].min().date()} -> {data['date'].max().date()}")

# ==========================================================
# CHRONOLOGICAL TRAIN/TEST SPLIT (by date, all symbols)
# ==========================================================
unique_dates = np.sort(data["date"].unique())
test_cutoff = pd.Timestamp(unique_dates[int(len(unique_dates) * (1 - TEST_FRACTION))])
train_mask, test_mask = date_split(data["date"], test_cutoff, HORIZON)
train, test = data[train_mask], data[test_mask]

X_train, y_train = train[FEATURE_COLS], train["target"]
X_test, y_test = test[FEATURE_COLS], test["target"]
train_end = train["date"].max()

print(f"[SPLIT] Train: {len(train)} rows ({train['date'].min().date()} -> {train_end.date()})")
print(f"[SPLIT] Test:  {len(test)} rows ({test_cutoff.date()} -> {test['date'].max().date()})")
print("[SPLIT] Test rows per symbol:", test["symbol"].value_counts().to_dict())
print(f"[SPLIT] UP ratio  train={y_train.mean():.3f}  test={y_test.mean():.3f}")

# ==========================================================
# WALK-FORWARD CV (by date, within the training period)
# ==========================================================
print(f"\n[CV] {CV_FOLDS}-fold walk-forward CV by date...")
train_dates = np.sort(train["date"].unique())
fold_edges = np.linspace(0, len(train_dates), CV_FOLDS + 2).astype(int)[1:]
cv_scores, cv_aucs = [], []
for i in range(CV_FOLDS):
    fold_start = pd.Timestamp(train_dates[fold_edges[i]])
    fold_end = pd.Timestamp(train_dates[min(fold_edges[i + 1], len(train_dates) - 1)])
    tr = train["date"] < fold_start - pd.Timedelta(days=HORIZON)
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

# ==========================================================
# FINAL TRAINING & EVALUATION
# ==========================================================
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

print("\n" + "=" * 60)
print("[EVAL] OUT-OF-SAMPLE TEST SET")
print("=" * 60)
print(f"  Baseline (majority class): {baseline_accuracy:.3f}")
print(f"  Accuracy:  {acc:.3f}  ({acc - baseline_accuracy:+.3f} vs baseline)")
print(f"  Precision: {precision:.3f}")
print(f"  Recall:    {recall:.3f}")
print(f"  F1-Score:  {f1:.3f}")
print(f"  ROC-AUC:   {roc_auc:.3f}  (0.5 = random)")
print(f"  Mean prob {probs.mean():.3f}, std {probs.std():.3f}, "
      f">0.5: {(probs > 0.5).mean():.1%}, >0.6: {(probs > 0.6).mean():.1%}")
print(f"  Confusion matrix [[TN, FP], [FN, TP]]: {cm.tolist()}")

per_symbol = {}
for sym, g in test.groupby("symbol"):
    p = probs[(test["symbol"] == sym).values]
    per_symbol[sym] = {
        "n": int(len(g)),
        "accuracy": float(accuracy_score(g["target"], (p > 0.5).astype(int))),
        "roc_auc": float(roc_auc_score(g["target"], p)) if g["target"].nunique() > 1 else None,
    }
    print(f"   {sym:10s} n={len(g):4d}  acc={per_symbol[sym]['accuracy']:.3f}  "
          f"auc={per_symbol[sym]['roc_auc'] if per_symbol[sym]['roc_auc'] is not None else float('nan'):.3f}")

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
    "trained_at": datetime.now().isoformat(timespec="seconds"),
    "symbols": COINS,
    "horizon": HORIZON,
    "feature_cols": FEATURE_COLS,
    "train_start": str(train["date"].min().date()),
    "train_end": str(train_end.date()),
    "test_start": str(test_cutoff.date()),
    "test_end": str(test["date"].max().date()),
}

# ==========================================================
# SAVE ARTIFACTS
# ==========================================================
joblib.dump(model, f"{ARTIFACTS}/general_crypto_model.pkl")
joblib.dump(list(FEATURE_COLS), f"{ARTIFACTS}/feature_columns.pkl")
joblib.dump(eval_metrics, f"{ARTIFACTS}/evaluation_metrics.pkl")
joblib.dump(metadata, f"{ARTIFACTS}/training_metadata.pkl")
if hasattr(model, "feature_importances_"):
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    joblib.dump(imp.to_dict(), f"{ARTIFACTS}/feature_importance.pkl")
    print("[FEATURES] Importance:\n" + imp.round(4).to_string())

print(f"\n[SAVE] Artifacts written to {ARTIFACTS}/ "
      "(model, feature_columns, evaluation_metrics, feature_importance, training_metadata)")
print("[DONE] Training complete!")
