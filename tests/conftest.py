"""Shared test setup: make the app importable and keep every test offline (no real API calls)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src", ROOT / "graph"):
    sys.path.insert(0, str(p))

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")  # the client needs *a* key to exist
os.environ.pop("GITHUB_DISPATCH_TOKEN", None)                    # tests must never send a real alert
