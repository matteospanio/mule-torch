"""Console-script entry points (thin wrappers around scripts/convert.py + verify.py).

These let ``mule-torch-convert`` / ``mule-torch-verify`` work after ``pip install``,
delegating to the ``scripts/`` modules which are also runnable directly with
``python scripts/convert.py ...``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _run(name: str) -> None:
    target = _SCRIPTS / name
    if not target.exists():
        raise SystemExit(f"script not found: {target}")
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


def convert_main() -> None:
    _run("convert.py")


def verify_main() -> None:
    _run("verify.py")
