"""Shared helpers for DIA training demo scripts (local save, optional Hub push)."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_impact_accounting.producer.tracking import track


def hub_push_enabled(repo: str) -> bool:
    """Return False when the run should stay local-only."""
    if not repo.strip():
        return False
    for var in ("DIA_LOCAL", "DIA_NO_PUSH"):
        if os.getenv(var, "").lower() in ("1", "true", "yes"):
            return False
    return True


def finalize_run(
    t: track,
    *,
    out_dir: str,
    repo: str,
    token: str | None,
    base_model: str,
    save_fn: Callable[[], None],
    push_fn: Callable[[], None] | None = None,
    interrupted: bool = False,
    dashboard_hint: str | None = None,
) -> int:
    """Save artifacts locally, write ``dia_report``, optionally push to the Hub."""
    if interrupted:
        print("\nInterrupted — finalizing partial run locally.")

    print(t.checklist_line())

    save_fn()
    card_path = t.write(os.path.join(out_dir, "README.md"))
    print(f"Saved artifacts under {out_dir}/")
    print(f"DIA report on model card: {card_path}")
    print(f"Validate locally: dia validate {card_path}")

    hint = dashboard_hint or base_model

    if not hub_push_enabled(repo):
        if os.getenv("DIA_LOCAL", "").lower() in ("1", "true", "yes"):
            reason = "DIA_LOCAL set"
        elif os.getenv("DIA_NO_PUSH", "").lower() in ("1", "true", "yes"):
            reason = "DIA_NO_PUSH set"
        elif not repo.strip():
            reason = "REPO empty"
        else:
            reason = "Hub push disabled"
        print(f"Skipping Hub push ({reason} — local-only).")
        print(f"Dashboard base model: {hint}")
        return 130 if interrupted else 0

    if not token:
        print("No HF token — skipping Hub push. Local artifacts kept.")
        print(f"Dashboard base model: {hint}")
        return 130 if interrupted else 0

    if push_fn is None:
        print("Skipping Hub push (no push callback).")
        print(f"Dashboard base model: {hint}")
        return 130 if interrupted else 0

    try:
        push_fn()
    except Exception as exc:
        print(f"WARNING: Hub push failed ({exc}). Local artifacts kept under {out_dir}/")
        print(f"Dashboard base model: {hint}")
        return 130 if interrupted else 1

    print("Done. Check:", f"https://huggingface.co/{repo}")
    print(f"Dashboard base model: {hint}")
    return 130 if interrupted else 0


def exit_from_finalize(code: int) -> None:
    """Exit the process when finalize_run reported a non-zero status."""
    if code:
        sys.exit(code)
