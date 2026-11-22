"""
config.py
Loads config.toml from the project root. Import CFG (the whole file as a dict) or the
named sections below.
"""

import os
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config.toml")

with open(CONFIG_PATH, "rb") as fh:
    CFG = tomllib.load(fh)

UNIVERSES = CFG["universes"]
UNIVERSES_LARGE = CFG["universes_large"]
MODEL = CFG["model"]
RESEARCH = CFG["research"]
BACKTEST = CFG["backtest"]
