"""Run doctests across ``ai_impact_accounting`` with proper package imports.

``python -m doctest`` loads each file as a standalone module, which breaks
relative imports in a src-layout package. This script imports modules by
package name instead and skips entrypoints that run side effects at import.
"""

from __future__ import annotations

import doctest
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Space entrypoint: initializes Store + WebhooksServer at import time.
_SKIP = frozenset({"ai_impact_accounting.dashboard.app"})


def main() -> int:
    """Discover package modules and run embedded doctests.

    Returns
    -------
    int
        ``0`` if all doctests pass, ``1`` otherwise.
    """
    failed = 0
    pkg_root = ROOT / "src" / "ai_impact_accounting"
    for path in sorted(pkg_root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        modname = ".".join(path.relative_to(ROOT / "src").with_suffix("").parts)
        if modname in _SKIP:
            continue
        mod = importlib.import_module(modname)
        result = doctest.testmod(mod, verbose=False, optionflags=doctest.NORMALIZE_WHITESPACE)
        if result.failed:
            print(f"{modname}: {result.failed} doctest(s) failed")
            failed += result.failed
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
