#!/usr/bin/env python3
"""Convenience entry: run single demo from project root."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import main

if __name__ == "__main__":
    raise SystemExit(main())
