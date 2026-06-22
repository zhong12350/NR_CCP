#!/usr/bin/env python3
"""Generate advisor demo figures (5 fields × 3 methods)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import main

if __name__ == "__main__":
    raise SystemExit(main(["advisor"]))
