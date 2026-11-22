"""
refresh.py
One command to bring everything up to date: download fresh prices and non-price data,
retrain the direction and big-move models, and optionally rerun the research.

Run from the project root:
    python -m src.refresh                # data + both models
    python -m src.refresh --research     # also rerun the research grid (about 5 minutes more)
    python -m src.refresh --data-only    # just refresh cached data

Schedule it (Windows Task Scheduler, every weekday at 07:00):
    schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 07:00 /TN "CryptoStockML refresh"
             /TR "cmd /c cd /d <project folder> && venv\\Scripts\\python.exe -m src.refresh >> refresh.log 2>&1"
"""

import argparse
import time
from datetime import datetime

from src import alt_data, research, train_model, vol_model
from src.config import MODEL
from src.model_registry import TRAINING_UNIVERSES


def refresh_data(classes) -> None:
    hours = MODEL["fresh_hours"]
    symbols = sorted({s for ac in classes for s in TRAINING_UNIVERSES[ac]} | set(research.CONTEXT_SYMBOLS.values()))
    print(f"[DATA] Prices ({research.LOOKBACK}, older than {hours}h are re-downloaded): {len(symbols)} symbols")
    for sym in symbols:
        df = research._load(sym, fresh_hours=hours)
        last = df.index.max().date() if not df.empty else "no data"
        stale = " (download failed, stale cache)" if df.attrs.get("stale_cache") else ""
        print(f"   {sym:9s} last bar {last}{stale}")

    print("[DATA] Non-price data")
    if "crypto" in classes:
        alt_data.fear_greed()
        for coin in ["BTC", "ETH"]:
            alt_data.deribit_funding(coin)
        for sym in TRAINING_UNIVERSES["crypto"]:
            alt_data.hyperliquid_funding(sym.split("-")[0])
    if "stock" in classes:
        for series in ["T10Y2Y", "BAA10Y"]:
            alt_data.fred(series)
        for sym in TRAINING_UNIVERSES["stock"]:
            alt_data.earnings(sym)
            alt_data.analyst_actions(sym)
    print("   done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=list(TRAINING_UNIVERSES), choices=list(TRAINING_UNIVERSES))
    ap.add_argument("--research", action="store_true", help="also rerun the research grid")
    ap.add_argument("--data-only", action="store_true", help="refresh data, skip retraining")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[REFRESH] {datetime.now().isoformat(timespec='seconds')}")
    refresh_data(args.classes)
    if not args.data_only:
        for ac in args.classes:
            train_model.train_asset_class(ac)
            vol_model.train_vol(ac)
    if args.research:
        res, rot = research.run(["logreg", "hgb", "rf"], args.classes)
        print(f"[RESEARCH] {research.write_report(res, rot)}")
    print(f"[REFRESH] done in {(time.time() - t0) / 60:.1f} min")
