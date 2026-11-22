import pandas as pd

from src.config import CFG
from src.data_loader import _read_cache
from src.model_registry import TRAINING_UNIVERSES, asset_class_for


def test_asset_class_for():
    assert asset_class_for("BTC-USD") == "crypto"
    assert asset_class_for(" eth-usd ") == "crypto"
    assert asset_class_for("AAPL") == "stock"
    assert asset_class_for("BRK-B") == "stock"


def test_config_universes_loaded():
    assert TRAINING_UNIVERSES["crypto"] == CFG["universes"]["crypto"]
    assert all(s.endswith("-USD") for s in CFG["universes_large"]["crypto"])
    assert not any(s.endswith("-USD") for s in CFG["universes_large"]["stock"])


def test_read_cache_drops_junk_rows(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("time,open,high,low,close,volume\n"
                 ",BTC-USD,BTC-USD,BTC-USD,BTC-USD,BTC-USD\n"
                 "2024-01-01,1,2,0.5,1.5,100\n"
                 "2024-01-02,1.5,2.5,1,2,200\n")
    df = _read_cache(str(p))
    assert len(df) == 2
    assert df["close"].tolist() == [1.5, 2.0]
    assert isinstance(df.index, pd.DatetimeIndex)
