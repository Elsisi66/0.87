"""Compatibility package shim so `python -m bot087...` resolves to `src/bot087` modules."""
from __future__ import annotations

from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_src_pkg = _pkg_dir.parent / "src" / "bot087"

# Allow importing submodules from src/bot087 without packaging install.
__path__ = [str(_pkg_dir)]
if _src_pkg.exists():
    __path__.append(str(_src_pkg))
