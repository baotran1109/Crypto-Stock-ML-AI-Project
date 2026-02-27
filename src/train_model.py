"""
train_model.py
Enhanced version: adds new indicators, 5-day target, time-series CV, feature importance, and optional XGBoost model.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
from src.data_loader import load_crypto
from src.features import add_indicators, make_supervised

# ==========================================================
# CONFIG
# ==========================================================
os.makedirs("artifacts", exist_ok=True)
coins = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "MATIC-USD"]
USE_XGB = False  # toggle to True if you want to use XGBoost instead of RF
FORCE_REFRESH = False  # Set to True to fetch fresh data from API (ignores cache)

if USE_XGB:
    from xgboost import XGBClassifier

# ==========================================================
# COLLECT & PREPARE DATA
# ==========================================================
frames = []

for sym in coins:
    print(f"\n[FETCH] Fetching {sym} ...")
    try:
        # Check if cache exists and age
        cache_path = f"data/{sym.replace('-', '_').upper()}__yahoo__5y.csv"
        from os.path import exists, getmtime
        from datetime import datetime
        if exists(cache_path) and not FORCE_REFRESH:
            cache_age_hours = (datetime.now().timestamp() - getmtime(cache_path)) / 3600
            print(f"   Using cached data (age: {cache_age_hours:.2f} hours)")
            print(f"   (Set FORCE_REFRESH=True to fetch from API)")
        else:
            if FORCE_REFRESH:
                print(f"   Force refresh enabled - fetching from API...")
            else:
                print(f"   No cache found - fetching from API...")
        
        df = load_crypto(sym, force_refresh=FORCE_REFRESH)
        if df.empty:
            print(f"[WARN] Skipping {sym}: no data loaded.")
            continue
        
        data_source = df['source'].iloc[-1] if 'source' in df.columns else 'unknown'
        latest_date = df.index.max()
        data_status = "API" if FORCE_REFRESH or not exists(cache_path) else "CACHE"
        print(f"   Loaded {len(df)} rows from {data_source} ({data_status}) (latest: {latest_date.date()})")
        
        df = add_indicators(df)
        if df.empty:
            print(f"[WARN] Skipping {sym}: dataframe empty after adding indicators.")
            # Try with force_refresh
            print(f"   Attempting with force_refresh=True...")
            df = load_crypto(sym, force_refresh=True)
            if not df.empty:
                df = add_indicators(df)
                if df.empty:
                    print(f"[WARN] Still empty after force_refresh. Skipping {sym}.")
                    continue
            else:
                continue
        
        if len(df) < 60:
            print(f"[WARN] Skipping {sym}: only {len(df)} rows after indicators (need at least 60).")
            continue
        
        print(f"   {len(df)} rows after adding indicators")

        X, y = make_supervised(df)
        if X.empty or y.empty:
            print(f"[WARN] Skipping {sym}: supervised dataset is empty (X: {len(X)}, y: {len(y)}).")
            continue
        
        print(f"   Supervised dataset: {len(X)} samples")

        X["symbol"] = sym
        df_supervised = pd.concat([X, y.rename("target")], axis=1).dropna()
        if df_supervised.empty:
            print(f"[WARN] Skipping {sym}: no valid samples after combining features and target.")
            continue
        
        frames.append(df_supervised)
        print(f"[OK] {sym}: {len(df_supervised)} samples ready.")
    except Exception as e:
        print(f"[ERROR] Error processing {sym}: {e}")
        import traceback
        traceback.print_exc()

if not frames:
    raise RuntimeError("[ERROR] No valid data to train on. Try force_refresh=True or check your feature pipeline.")

data = pd.concat(frames).dropna()
data = pd.get_dummies(data, columns=["symbol"])
print(f"\n[DATA] Combined dataset size: {len(data)} rows, {len(data.columns)} columns")

# ==========================================================
# TRAIN/TEST SPLIT
# ==========================================================
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]
X_train, y_train = train.drop("target", axis=1), train["target"]
X_test, y_test = test.drop("target", axis=1), test["target"]

print(f"[SPLIT] Training set: {len(X_train)} | Test set: {len(X_test)}")

# Class distribution analysis
print(f"\n[ANALYSIS] Class Distribution:")
train_pos_ratio = y_train.mean()
test_pos_ratio = y_test.mean()
print(f"  Training set - Positive class (UP): {train_pos_ratio:.3f} ({train_pos_ratio*100:.1f}%)")
print(f"  Training set - Negative class (DOWN): {1-train_pos_ratio:.3f} ({(1-train_pos_ratio)*100:.1f}%)")
print(f"  Test set - Positive class (UP): {test_pos_ratio:.3f} ({test_pos_ratio*100:.1f}%)")
print(f"  Test set - Negative class (DOWN): {1-test_pos_ratio:.3f} ({(1-test_pos_ratio)*100:.1f}%)")

if abs(train_pos_ratio - 0.5) > 0.1:
    print(f"\n[WARN] Class imbalance detected! UP/DOWN ratio is {train_pos_ratio:.3f} - model may be biased.")

# ==========================================================
# MODEL SELECTION
# ==========================================================
if USE_XGB:
    model = XGBClassifier(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
    )
else:
    # Calculate class weights for better balancing
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    print(f"\n[MODEL] Using class weights: {class_weight_dict}")
    
    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=8,
        min_samples_leaf=5,
        class_weight=class_weight_dict,  # Use explicit class weights instead of balanced_subsample
        n_jobs=-1,
        random_state=42,
    )

# ==========================================================
# TIME SERIES CROSS-VALIDATION
# ==========================================================
print("\n[CV] Performing 5-fold TimeSeriesSplit CV...")
tscv = TimeSeriesSplit(n_splits=5)
scores = []
for tr_idx, te_idx in tscv.split(X_train):
    m = model.__class__(**model.get_params())
    m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
    score = m.score(X_train.iloc[te_idx], y_train.iloc[te_idx])
    scores.append(score)
print(f"[CV] Average CV accuracy: {np.mean(scores):.3f}")

# ==========================================================
# FINAL TRAINING
# ==========================================================
print("\n[TRAIN] Training final model...")
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# ==========================================================
# COMPREHENSIVE EVALUATION METRICS
# ==========================================================
print("\n" + "="*60)
print("[EVAL] COMPREHENSIVE MODEL EVALUATION")
print("="*60)

# Baseline comparison
baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())
print(f"\n[EVAL] Baseline (always predict majority class): {baseline_accuracy:.3f}")
print(f"[EVAL] Model Accuracy: {acc:.3f}")
print(f"[EVAL] Improvement over baseline: {acc - baseline_accuracy:.3f} ({((acc - baseline_accuracy) / baseline_accuracy * 100):.1f}%)")

# Detailed metrics
precision = precision_score(y_test, preds, zero_division=0)
recall = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)

print(f"\n[EVAL] Classification Metrics:")
print(f"  Precision: {precision:.3f}")
print(f"  Recall:    {recall:.3f}")
print(f"  F1-Score:  {f1:.3f}")

# ROC-AUC if probabilities available
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probs)
    print(f"  ROC-AUC:   {roc_auc:.3f}")
    
    # Prediction distribution analysis
    print(f"\n[ANALYSIS] Prediction Distribution:")
    prob_mean = probs.mean()
    prob_std = probs.std()
    prob_median = np.median(probs)
    up_predictions = (probs > 0.5).sum()
    down_predictions = (probs <= 0.5).sum()
    strong_up = (probs > 0.7).sum()
    strong_down = (probs < 0.3).sum()
    
    print(f"  Mean probability: {prob_mean:.3f}")
    print(f"  Median probability: {prob_median:.3f}")
    print(f"  Std deviation: {prob_std:.3f}")
    print(f"  Predictions > 0.5 (UP): {up_predictions} ({up_predictions/len(probs)*100:.1f}%)")
    print(f"  Predictions <= 0.5 (DOWN): {down_predictions} ({down_predictions/len(probs)*100:.1f}%)")
    print(f"  Strong UP (>0.7): {strong_up} ({strong_up/len(probs)*100:.1f}%)")
    print(f"  Strong DOWN (<0.3): {strong_down} ({strong_down/len(probs)*100:.1f}%)")
    
    if up_predictions / len(probs) > 0.7:
        print(f"  [WARN] Model heavily biased toward UP predictions!")
    elif down_predictions / len(probs) > 0.7:
        print(f"  [WARN] Model heavily biased toward DOWN predictions!")
    
    # Threshold tuning
    th = 0.55
    preds_th = (probs > th).astype(int)
    acc_custom = accuracy_score(y_test, preds_th)
    precision_th = precision_score(y_test, preds_th, zero_division=0)
    recall_th = recall_score(y_test, preds_th, zero_division=0)
    print(f"\n[EVAL] Metrics @ {th:.2f} confidence threshold:")
    print(f"  Accuracy:  {acc_custom:.3f}")
    print(f"  Precision: {precision_th:.3f}")
    print(f"  Recall:    {recall_th:.3f}")
else:
    roc_auc = None
    probs = None

# Confusion Matrix
cm = confusion_matrix(y_test, preds)
print(f"\n[EVAL] Confusion Matrix:")
print(f"  True Negatives:  {cm[0,0]:4d}  |  False Positives: {cm[0,1]:4d}")
print(f"  False Negatives: {cm[1,0]:4d}  |  True Positives:  {cm[1,1]:4d}")

# Classification Report
print(f"\n[EVAL] Detailed Classification Report:")
print(classification_report(y_test, preds, target_names=['DOWN', 'UP']))

# Save evaluation metrics
eval_metrics = {
    'accuracy': float(acc),
    'baseline_accuracy': float(baseline_accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'roc_auc': float(roc_auc) if roc_auc is not None else None,
    'confusion_matrix': cm.tolist(),
    'test_size': len(y_test),
    'positive_class_ratio': float(y_test.mean())
}
joblib.dump(eval_metrics, "artifacts/evaluation_metrics.pkl")
print(f"\n[SAVE] Evaluation metrics saved to artifacts/evaluation_metrics.pkl")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================
if hasattr(model, "feature_importances_"):
    imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\n[FEATURES] Top 15 features by importance:")
    print(imp.head(15))

# ==========================================================
# SAVE ARTIFACTS
# ==========================================================
joblib.dump(model, "artifacts/general_crypto_model.pkl")
joblib.dump(list(X_train.columns), "artifacts/feature_columns.pkl")

# Save feature importance for visualization
if hasattr(model, "feature_importances_"):
    feature_importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    joblib.dump(feature_importance.to_dict(), "artifacts/feature_importance.pkl")

print("\n[SAVE] Artifacts saved in /artifacts:")
print(" - general_crypto_model.pkl")
print(" - feature_columns.pkl")
print(" - evaluation_metrics.pkl")
if hasattr(model, "feature_importances_"):
    print(" - feature_importance.pkl")
print("[DONE] Training complete!")
