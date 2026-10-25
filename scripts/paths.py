"""Project paths. Importing this module also makes the `pacr` package importable."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PACR_DATA", ROOT / "data"))
FEATURES = Path(os.environ.get("PACR_FEATURES", ROOT / "features"))
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = ROOT / "figures"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
