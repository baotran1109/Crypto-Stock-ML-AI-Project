"""
research.py
Signal research harness: tests many (target x feature set x model) combinations and
scores each one honestly on data it has never seen.

How it scores
- Walk-forward by DATE: the last TEST_FRACTION of dates is cut into N_FOLDS blocks.
  Each block is predicted by a model trained only on dates before it, with a purge gap
  of `horizon` bars so no training label overlaps the test block.
- ROC-AUC per fold (0.5 = random). We report the mean, the worst fold, and how many
  folds beat 0.5, because one lucky period is not a signal.
- Spread: average outcome of the top 20% of predictions minus the bottom 20%,
  i.e. "how much better were the trades the model liked most?"
- Every target also gets a no-ML "rule" row (raw 20-day momentum for direction
  targets, raw 20-day volatility for the big-move target). A model has to beat its rule.

Crypto and stocks are researched separately, each with its own context features.

Run from the project root:
    python -m src.research                 # full grid
    python -m src.research --quick         # fewer models, for a fast check
    python -m src.research --no-rotation   # skip the crypto rotation simulation
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_crypto
from src.features import FEATURE_COLS, add_indicators
from src.model_registry import TRAINING_UNIVERSES

warnings.filterwarnings("ignore", category=RuntimeWarning)

LOOKBACK = "10y"
TEST_FRACTION = 0.6
N_FOLDS = 6
MIN_XS_SYMBOLS = 4
REPORT_DIR = "reports"

UNIVERSES = TRAINING_UNIVERSES  # same symbols the production models train on
CONTEXT_SYMBOLS = {"vix": "^VIX", "tnx": "^TNX", "dxy": "DX-Y.NYB", "spy": "SPY", "btc": "BTC-USD"}

PRICE_EXTRA = ["ret60", "high252_dist", "vol60", "vol20_60"]
VOLUME = ["volu_z20", "volu_trend"]
CROSS = {
    "crypto": ["btc_ret5", "btc_ret20", "btc_vol20", "rel20_btc", "vix_level", "vix_chg5", "spy_ret20"],
    "stock": ["spy_ret5", "spy_ret20", "rel20_spy", "vix_level", "vix_chg5", "tnx_chg20", "dxy_ret20"],
}
XS = ["xs_ret5", "xs_ret20", "xs_ret60"]
CALENDAR = ["dow"]


def feature_sets(asset_class: str) -> Dict[str, List[str]]:
    base = list(FEATURE_COLS)
    price = base + PRICE_EXTRA
    volume = price + VOLUME
    cross = volume + CROSS[asset_class]
    return {
        "base": base,
        "+price": price,
        "+volume": volume,
        "+cross": cross,
        "all": cross + XS + CALENDAR,
    }


# target name -> (label column, horizon bars, outcome column used for the spread)
TARGETS = {
    "dir5": ("y_dir5", 5, "fwd5"),
    "dir20": ("y_dir20", 20, "fwd20"),
    "excess5": ("y_excess5", 5, "excess5"),
    "bigmove5": ("y_big5", 5, "absfwd5"),
}


# ==========================================================
# DATA
# ==========================================================
def _load(symbol: str) -> pd.DataFrame:
    try:
        return load_crypto(symbol, lookback=LOOKBACK, prefer="yahoo", fresh_hours=24 * 7)
    except Exception as e:
        print(f"   [WARN] {symbol}: {e}")
        return pd.DataFrame()


def load_context() -> pd.DataFrame:
    closes = {}
    for name, sym in CONTEXT_SYMBOLS.items():
        df = _load(sym)
        if not df.empty:
            closes[name] = df["close"]
    ctx = pd.DataFrame(closes).sort_index()
    out = pd.DataFrame(index=ctx.index)
    if "btc" in ctx:
        b = ctx["btc"].dropna()
        out["btc_ret5"] = b.pct_change(5)
        out["btc_ret20"] = b.pct_change(20)
        out["btc_vol20"] = b.pct_change().rolling(20).std()
    if "spy" in ctx:
        s = ctx["spy"].dropna()
        out["spy_ret5"] = s.pct_change(5)
        out["spy_ret20"] = s.pct_change(20)
    if "vix" in ctx:
        v = ctx["vix"].dropna()
        out["vix_level"] = v
        out["vix_chg5"] = v.pct_change(5)
    if "tnx" in ctx:
        out["tnx_chg20"] = ctx["tnx"].dropna().diff(20)
    if "dxy" in ctx:
        out["dxy_ret20"] = ctx["dxy"].dropna().pct_change(20)
    return out


def asset_frame(symbol: str) -> pd.DataFrame:
    raw = _load(symbol)
    if raw.empty:
        return pd.DataFrame()
    df = add_indicators(raw)
    if df.empty:
        return pd.DataFrame()
    c = df["close"]
    r = c.pct_change()
    f = df[FEATURE_COLS].copy()

    f["ret60"] = c.pct_change(60)
    f["high252_dist"] = c / c.rolling(252, min_periods=60).max() - 1
    f["vol60"] = r.rolling(60).std()
    f["vol20_60"] = df["vol20"] / f["vol60"]

    v = df["volume"].replace(0, np.nan)
    f["volu_z20"] = np.log(v / v.rolling(20).mean())
    f["volu_trend"] = np.log(v.rolling(5).mean() / v.rolling(20).mean())
    f["dow"] = df.index.dayofweek

    # outcomes (future values: used ONLY as labels, never as features)
    f["fwd5"] = c.shift(-5) / c - 1
    f["fwd10"] = c.shift(-10) / c - 1
    f["fwd20"] = c.shift(-20) / c - 1
    f["absfwd5"] = f["fwd5"].abs()
    big_thr = (c / c.shift(5) - 1).abs().rolling(252, min_periods=60).median()  # known at t
    f["y_big5"] = np.where(f["fwd5"].notna() & big_thr.notna(),
                           (f["absfwd5"] > big_thr).astype(float), np.nan)
    f["y_dir5"] = np.where(f["fwd5"].notna(), (f["fwd5"] > 0).astype(float), np.nan)
    f["y_dir20"] = np.where(f["fwd20"].notna(), (f["fwd20"] > 0).astype(float), np.nan)

    f["symbol"] = symbol
    f["date"] = f.index
    return f.reset_index(drop=True)


def build_panel(asset_class: str, context: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for sym in UNIVERSES[asset_class]:
        f = asset_frame(sym)
        if f.empty:
            continue
        print(f"   {sym:9s} {len(f):5d} rows  {f['date'].min().date()} -> {f['date'].max().date()}")
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)

    # context features: last known value on or before each date (ffill = no look-ahead)
    ctx = context.reindex(context.index.union(panel["date"].unique())).sort_index().ffill()
    ctx = ctx.loc[~ctx.index.duplicated()]
    panel = panel.join(ctx, on="date")
    if asset_class == "crypto":
        panel["rel20_btc"] = panel["ret20"] - panel["btc_ret20"]
    else:
        panel["rel20_spy"] = panel["ret20"] - panel["spy_ret20"]

    # cross-sectional: rank vs the other symbols on the same date
    g = panel.groupby("date")
    count = g["close_ema12"].transform("count")
    for col, src in [("xs_ret5", "ret5"), ("xs_ret20", "ret20"), ("xs_ret60", "ret60")]:
        panel[col] = g[src].rank(pct=True).where(count >= MIN_XS_SYMBOLS)
    med_fwd5 = g["fwd5"].transform("median")
    panel["excess5"] = panel["fwd5"] - med_fwd5
    panel["y_excess5"] = np.where(panel["fwd5"].notna() & (count >= MIN_XS_SYMBOLS),
                                  (panel["excess5"] > 0).astype(float), np.nan)

    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=FEATURE_COLS).sort_values(["date", "symbol"]).reset_index(drop=True)
    return panel


# ==========================================================
# MODELS
# ==========================================================
def make_model(name: str):
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=2000))
    if name == "hgb":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                              min_samples_leaf=200, l2_regularization=1.0,
                                              random_state=0)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=100,
                                      max_features="sqrt", n_jobs=-1, random_state=0)
    raise ValueError(name)


def rule_score(df: pd.DataFrame, target: str) -> np.ndarray:
    return (df["vol20"] if target == "bigmove5" else df["ret20"]).values


# ==========================================================
# WALK-FORWARD EVALUATION
# ==========================================================
def walk_forward(panel: pd.DataFrame, target: str, features: List[str], model_name: str) -> Dict:
    ycol, horizon, outcome = TARGETS[target]
    data = panel.dropna(subset=[ycol])
    dates = np.sort(data["date"].unique())
    first_test = int(len(dates) * (1 - TEST_FRACTION))
    edges = np.linspace(first_test, len(dates), N_FOLDS + 1).astype(int)

    aucs, spreads, fold_info = [], [], []
    for k in range(N_FOLDS):
        lo, hi = edges[k], edges[k + 1]
        test_dates = dates[lo:hi]
        train_end = dates[max(lo - horizon - 1, 0)]
        tr = data[data["date"] < train_end]
        te = data[data["date"].isin(test_dates)]
        if len(tr) < 500 or te[ycol].nunique() < 2:
            continue

        if model_name == "rule":
            score = rule_score(te, target)
            ok = ~np.isnan(score)
            te, score = te[ok], score[ok]
        else:
            m = make_model(model_name)
            m.fit(tr[features], tr[ycol].astype(int))
            score = m.predict_proba(te[features])[:, 1]

        auc = roc_auc_score(te[ycol].astype(int), score)
        q_hi, q_lo = np.quantile(score, 0.8), np.quantile(score, 0.2)
        spread = te.loc[score >= q_hi, outcome].mean() - te.loc[score <= q_lo, outcome].mean()
        aucs.append(auc)
        spreads.append(spread)
        fold_info.append(f"{pd.Timestamp(test_dates[0]).date()}:{auc:.3f}")

    aucs = np.array(aucs)
    return {
        "mean_auc": float(aucs.mean()) if len(aucs) else np.nan,
        "min_auc": float(aucs.min()) if len(aucs) else np.nan,
        "folds_above_50": int((aucs > 0.5).sum()),
        "n_folds": int(len(aucs)),
        "mean_spread": float(np.mean(spreads)) if spreads else np.nan,
        "folds": " ".join(fold_info),
        "test_start": str(pd.Timestamp(dates[first_test]).date()),
        "test_end": str(pd.Timestamp(dates[-1]).date()),
        "n_rows": int(len(data)),
    }


def run(models: List[str], classes: List[str], rotation: bool = True):
    print("[DATA] Context series")
    context = load_context()
    rows = []
    rotation_df = None
    for ac in classes:
        print(f"\n[DATA] {ac} universe ({LOOKBACK})")
        panel = build_panel(ac, context)
        print(f"   panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols")
        fsets = feature_sets(ac)
        for target in TARGETS:
            res = walk_forward(panel, target, [], "rule")
            rows.append({"asset_class": ac, "target": target, "features": "-", "model": "rule", **res})
            print(f"   {ac:6s} {target:8s} {'-':8s} {'rule':6s} auc={res['mean_auc']:.3f} "
                  f"min={res['min_auc']:.3f} folds>.5={res['folds_above_50']}/{res['n_folds']}")
            for fs_name, cols in fsets.items():
                for mname in models:
                    if mname == "rf" and fs_name not in ("base", "all"):
                        continue  # RF is slow: only run it on the two ends of the ladder
                    t0 = time.time()
                    res = walk_forward(panel, target, cols, mname)
                    rows.append({"asset_class": ac, "target": target, "features": fs_name,
                                 "model": mname, **res})
                    print(f"   {ac:6s} {target:8s} {fs_name:8s} {mname:6s} auc={res['mean_auc']:.3f} "
                          f"min={res['min_auc']:.3f} folds>.5={res['folds_above_50']}/{res['n_folds']} "
                          f"spread={res['mean_spread']:+.4f} ({time.time() - t0:.0f}s)")
        if rotation and ac == "crypto":
            print("\n[ROTATION] crypto")
            rotation_df = run_rotation(panel, ac)
    return pd.DataFrame(rows), rotation_df


# ==========================================================
# ROTATION SIMULATION  (does the excess5 ranking make money after costs?)
# ==========================================================
def _rotation_stats(period_rets: np.ndarray, periods_per_year: float) -> Dict:
    equity = np.cumprod(1 + period_rets)
    peak = np.maximum.accumulate(equity)
    sd = period_rets.std()
    return {
        "total_return": float(equity[-1] - 1),
        "cagr": float(equity[-1] ** (periods_per_year / len(period_rets)) - 1),
        "sharpe": float(period_rets.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0,
        "max_drawdown": float(((peak - equity) / peak).max()),
    }


def rotation_test(panel: pd.DataFrame, features: List[str], model_name: str,
                  top_k: int = 3, cost: float = 0.0015, hold: int = 5,
                  n_random: int = 200, seed: int = 0) -> pd.DataFrame:
    """
    Walk-forward rotation: on every `hold`-th date of each test fold, hold the top_k symbols
    by predicted P(outperform the median), equal weight, for `hold` bars. Costs are charged
    on turnover (sum of |weight changes|) at `cost` per unit traded.
    Compared against: equal-weight basket of all symbols, top_k by 20-day momentum,
    and top_k picked at random (n_random runs).
    hold must be 5, 10 or 20 (the forward-return columns that exist).
    """
    fwd_col = f"fwd{hold}"
    assert fwd_col in panel.columns, f"no {fwd_col} column"
    ycol, horizon, _ = TARGETS["excess5"]
    data = panel.dropna(subset=[ycol])
    dates = np.sort(data["date"].unique())
    first_test = int(len(dates) * (1 - TEST_FRACTION))
    edges = np.linspace(first_test, len(dates), N_FOLDS + 1).astype(int)
    rng = np.random.default_rng(seed)

    # score every test row with a model trained only on earlier data (same folds as walk_forward)
    scored = []
    for k in range(N_FOLDS):
        lo, hi = edges[k], edges[k + 1]
        test_dates = dates[lo:hi]
        train_end = dates[max(lo - horizon - 1, 0)]
        tr = data[data["date"] < train_end]
        te = data[data["date"].isin(test_dates)].copy()
        m = make_model(model_name)
        m.fit(tr[features], tr[ycol].astype(int))
        te["score"] = m.predict_proba(te[features])[:, 1]
        scored.append(te)
    scored = pd.concat(scored)

    rebal_dates = np.sort(scored["date"].unique())[::hold]
    names = ["ML top-%d" % top_k, "Momentum top-%d" % top_k, "Equal-weight basket"]
    rets = {n: [] for n in names}
    prev_w = {n: {} for n in names}
    rand_rets = np.zeros((n_random, len(rebal_dates)))
    rand_prev = [dict() for _ in range(n_random)]

    def step(name_or_idx, weights, fwd, store_prev):
        prev = store_prev
        turnover = sum(abs(weights.get(s, 0) - prev.get(s, 0)) for s in set(weights) | set(prev))
        gross = sum(w * fwd[s] for s, w in weights.items())
        return gross - turnover * cost

    for j, d in enumerate(rebal_dates):
        day = scored[scored["date"] == d].dropna(subset=[fwd_col])
        if len(day) < top_k + 1:
            for n in names:
                rets[n].append(0.0)
            continue
        fwd = dict(zip(day["symbol"], day[fwd_col]))
        picks = {
            names[0]: day.nlargest(top_k, "score")["symbol"],
            names[1]: day.nlargest(top_k, "ret20")["symbol"],
            names[2]: day["symbol"],
        }
        for n, syms in picks.items():
            w = {s: 1 / len(syms) for s in syms}
            rets[n].append(step(n, w, fwd, prev_w[n]))
            prev_w[n] = w
        for r in range(n_random):
            syms = rng.choice(day["symbol"].values, size=top_k, replace=False)
            w = {s: 1 / top_k for s in syms}
            rand_rets[r, j] = step(r, w, fwd, rand_prev[r])
            rand_prev[r] = w

    ppy = 365 / hold if panel["symbol"].iloc[0].endswith("-USD") else 252 / hold
    rows = [{"strategy": n, **_rotation_stats(np.array(v), ppy)} for n, v in rets.items()]
    rand_stats = pd.DataFrame([_rotation_stats(rand_rets[r], ppy) for r in range(n_random)])
    rows.append({"strategy": f"Random top-{top_k} (median of {n_random})", **rand_stats.median().to_dict()})
    out = pd.DataFrame(rows)
    ml_total = out.loc[0, "total_return"]
    out.attrs["ml_beats_random_pct"] = float((ml_total > rand_stats["total_return"]).mean())
    out.attrs["period"] = f"{pd.Timestamp(rebal_dates[0]).date()} -> {pd.Timestamp(rebal_dates[-1]).date()}"
    out.attrs["n_periods"] = len(rebal_dates)
    return out


# Every rotation configuration that was tried is listed here and reported, winners and losers,
# so the report cannot cherry-pick. (feature set, model, top_k, cost per side, holding bars)
ROTATION_GRID = [
    ("+cross", "logreg", 3, 0.0015, 5), ("+cross", "logreg", 3, 0.0030, 5), ("+cross", "logreg", 5, 0.0015, 5),
    ("all", "hgb", 3, 0.0015, 5), ("all", "hgb", 3, 0.0030, 5), ("all", "hgb", 5, 0.0015, 5),
    ("base", "logreg", 3, 0.0015, 5), ("base", "logreg", 3, 0.0030, 5), ("base", "logreg", 5, 0.0015, 5),
    ("+cross", "logreg", 3, 0.0015, 10), ("+cross", "logreg", 5, 0.0015, 10),
    ("+cross", "logreg", 3, 0.0015, 20), ("+cross", "logreg", 5, 0.0015, 20),
]


def run_rotation(panel: pd.DataFrame, asset_class: str) -> pd.DataFrame:
    fsets = feature_sets(asset_class)
    rows = []
    for fs_name, model, top_k, cost, hold in ROTATION_GRID:
        out = rotation_test(panel, fsets[fs_name], model, top_k=top_k, cost=cost, hold=hold)
        by = out.set_index("strategy")
        ml, mom, basket = by.iloc[0], by.iloc[1], by.iloc[2]
        rows.append({
            "config": f"{fs_name}/{model} top-{top_k} hold {hold} cost {cost:.2%}",
            "ml_total": ml["total_return"], "ml_sharpe": ml["sharpe"], "ml_mdd": ml["max_drawdown"],
            "momentum_total": mom["total_return"], "basket_total": basket["total_return"],
            "random_median_total": by.iloc[3]["total_return"],
            "ml_beats_random": out.attrs["ml_beats_random_pct"],
            "period": out.attrs["period"],
        })
        print(f"   rotation {rows[-1]['config']:40s} ML {ml['total_return']:+.1%}  momentum {mom['total_return']:+.1%}  "
              f"basket {basket['total_return']:+.1%}  beats random {out.attrs['ml_beats_random_pct']:.0%}")
    return pd.DataFrame(rows)


def write_report(results: pd.DataFrame, rotation: Optional[pd.DataFrame] = None) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    results.to_csv(os.path.join(REPORT_DIR, "signal_research.csv"), index=False)
    lines = ["# Signal research results", "",
             f"Walk-forward by date, {N_FOLDS} folds over the last {int(TEST_FRACTION * 100)}% of dates, "
             f"purge gap = target horizon. ROC-AUC 0.5 = random. 'rule' = no-ML baseline.", ""]
    for (ac, target), g in results.groupby(["asset_class", "target"], sort=False):
        g = g.sort_values("mean_auc", ascending=False)
        lines += [f"## {ac} / {target}  (test {g['test_start'].iloc[0]} -> {g['test_end'].iloc[0]})", "",
                  "| features | model | mean AUC | worst fold | folds > 0.5 | top-bottom spread |",
                  "|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(f"| {r['features']} | {r['model']} | {r['mean_auc']:.3f} | {r['min_auc']:.3f} | "
                         f"{r['folds_above_50']}/{r['n_folds']} | {r['mean_spread']:+.2%} |")
        lines.append("")
    if rotation is not None and len(rotation):
        rotation.to_csv(os.path.join(REPORT_DIR, "rotation_research.csv"), index=False)
        lines += ["## crypto rotation (does the excess5 ranking make money after costs?)", "",
                  f"Every `hold` bars, hold the top-k coins by predicted P(beat the median coin), equal weight. "
                  f"Walk-forward, period {rotation['period'].iloc[0]}. Every configuration tried is listed.", "",
                  "| config | ML total | ML Sharpe | ML max DD | momentum top-k | equal-weight basket | random top-k (median) | ML beats random |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in rotation.iterrows():
            lines.append(f"| {r['config']} | {r['ml_total']:+.1%} | {r['ml_sharpe']:+.2f} | {r['ml_mdd']:.0%} | "
                         f"{r['momentum_total']:+.1%} | {r['basket_total']:+.1%} | {r['random_median_total']:+.1%} | "
                         f"{r['ml_beats_random']:.0%} |")
        n = len(rotation)
        lines += ["", f"ML beat the equal-weight basket in {(rotation['ml_total'] > rotation['basket_total']).sum()}/{n} "
                      f"configs and momentum in {(rotation['ml_total'] > rotation['momentum_total']).sum()}/{n}.", ""]
    path = os.path.join(REPORT_DIR, "signal_research.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="logreg + hgb only")
    ap.add_argument("--classes", nargs="+", default=list(UNIVERSES))
    ap.add_argument("--no-rotation", action="store_true", help="skip the crypto rotation simulation")
    args = ap.parse_args()
    models = ["logreg", "hgb"] if args.quick else ["logreg", "hgb", "rf"]
    t0 = time.time()
    res, rot = run(models, args.classes, rotation=not args.no_rotation)
    path = write_report(res, rot)
    print(f"\n[DONE] {len(res)} experiments in {(time.time() - t0) / 60:.1f} min -> {path}")
