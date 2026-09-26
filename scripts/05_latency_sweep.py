"""Step 5: full experiment suite (same as run_all.py)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_all

if __name__ == "__main__":
    run_all.main()
