"""Put product scripts and archived experiment runners on sys.path.

Experiment modules cross-import each other and the product evaluators by plain
module name, so both directories must be importable for the review-pipeline
tests.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "scripts", ROOT / "scripts" / "experiments"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
