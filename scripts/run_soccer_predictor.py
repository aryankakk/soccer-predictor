from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soccer_predictor.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
