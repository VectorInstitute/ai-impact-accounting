"""Local file-backed store for dashboard development and stress tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..models import Node, Report

STATE_FILE = "state.json"


class LocalStore:
    """In-memory accounting state loaded from a local ``state.json`` (no Hub I/O)."""

    def __init__(self, state_path: str | Path) -> None:
        self.path = Path(state_path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"State file not found: {self.path}")
        self.repo = f"local:{self.path.name}"
        self.token: Optional[str] = None
        self.nodes: dict[str, Node] = {}
        self.load()

    def load(self) -> None:
        """Load nodes from the configured state file."""
        with self.path.open(encoding="utf-8") as f:
            raw = json.load(f)
        nodes: dict[str, Node] = {}
        for mid, d in raw.get("nodes", {}).items():
            rep = Report.from_dict(d["report"]) if d.get("report") else None
            nodes[mid] = Node(
                model_id=mid,
                report=rep,
                lineage=d.get("lineage", []),
                updated_at=d.get("updated_at"),
            )
        self.nodes = nodes

    def save(self) -> None:
        """LocalStore is read-only — use the generator script to rewrite the file."""
        raise RuntimeError("LocalStore is read-only; regenerate state.json instead.")

    def upsert(self, node: Node, persist: bool = True) -> None:
        raise RuntimeError("LocalStore is read-only.")
