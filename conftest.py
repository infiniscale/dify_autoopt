"""Project-wide pytest configuration.

Ensures the repository root is on sys.path so imports like `from src...` work
reliably during test collection.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

