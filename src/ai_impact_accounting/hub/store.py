"""Persistence layer for accounting state.

Hugging Face Space filesystems are ephemeral, so the source of truth is a HF
*Dataset* repo holding ``state.json``. That also makes the accounting data itself
open and auditable. State is held in memory for fast reads and committed back to
the dataset on each upsert. For high write volume you would batch commits; for a
single-family demonstrator, commit-per-event is fine.

Each save also writes a flat ``nodes.parquet`` (or ``nodes.csv`` when pyarrow is
unavailable) so Hugging Face's Dataset Viewer can render a browsable table.
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import os
import threading
from datetime import UTC
from typing import Any, Optional

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from ..models import Node, Report


STATE_FILE = "state.json"
PARQUET_FILE = "nodes.parquet"
CSV_FILE = "nodes.csv"

_FLAT_COLUMNS = (
    "model_id",
    "has_report",
    "parent_models",
    "relation",
    "scope",
    "energy_kwh_lo",
    "energy_kwh_hi",
    "carbon_kgco2eq_lo",
    "carbon_kgco2eq_hi",
    "water_liters_lo",
    "water_liters_hi",
    "energy_quality",
    "carbon_quality",
    "water_quality",
    "gpu",
    "gpu_count",
    "gpu_hours",
    "region",
    "tool",
    "method",
    "updated_at",
)


def flatten_nodes(nodes: dict[str, Node]) -> list[dict[str, Any]]:
    """Return one flat row per tracked model for tabular export."""
    rows: list[dict[str, Any]] = []
    for mid in sorted(nodes):
        n = nodes[mid]
        row: dict[str, Any] = {
            "model_id": mid,
            "has_report": n.has_report,
            "parent_models": ",".join(e["model"] for e in n.lineage),
            "relation": ",".join(e.get("relation", "") for e in n.lineage),
            "updated_at": n.updated_at,
        }
        if n.report:
            r = n.report
            row.update(
                {
                    "scope": r.scope,
                    "energy_kwh_lo": r.energy.lo,
                    "energy_kwh_hi": r.energy.hi,
                    "carbon_kgco2eq_lo": r.carbon.lo,
                    "carbon_kgco2eq_hi": r.carbon.hi,
                    "water_liters_lo": r.water.lo,
                    "water_liters_hi": r.water.hi,
                    "energy_quality": r.quality.get("energy"),
                    "carbon_quality": r.quality.get("carbon"),
                    "water_quality": r.quality.get("water"),
                    "gpu": r.gpu,
                    "gpu_count": r.gpu_count,
                    "gpu_hours": r.gpu_hours,
                    "region": r.region,
                    "tool": r.tool,
                    "method": r.method,
                }
            )
        rows.append(row)
    return rows


def _encode_flat_export(rows: list[dict[str, Any]]) -> tuple[str, bytes]:
    """Serialize flat rows to Parquet, falling back to CSV without pyarrow."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(rows), buf)
        return PARQUET_FILE, buf.getvalue()
    except ImportError:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_FLAT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in _FLAT_COLUMNS})
        return CSV_FILE, buf.getvalue().encode()


class Store:
    """In-memory accounting state backed by a Hugging Face dataset repo.

    Parameters
    ----------
    dataset_repo : str
        Dataset repo id; a bare name is expanded to ``<user>/<name>``.
    token : str, optional
        Hugging Face write token; falls back to ``HF_TOKEN``.
    """

    def __init__(self, dataset_repo: str, token: Optional[str] = None) -> None:
        """Resolve the repo, ensure it exists, and load existing state."""
        self.token = token or os.getenv("HF_TOKEN")
        self.api = HfApi(token=self.token)
        self.repo = self._normalize_repo(dataset_repo)
        self._lock = threading.Lock()
        self.nodes: dict[str, Node] = {}
        self._ensure_repo()
        self.load()

    # ---- repo lifecycle ------------------------------------------------------
    def _normalize_repo(self, dataset_repo: str) -> str:
        """Expand a bare ``name`` to ``<user>/name`` (HF repo ids need a namespace)."""
        if "/" in dataset_repo:
            return dataset_repo
        user = self.api.whoami()["name"]
        return f"{user}/{dataset_repo}"

    def _ensure_repo(self) -> None:
        try:
            self.api.repo_info(self.repo, repo_type="dataset")
        except RepositoryNotFoundError:
            self.api.create_repo(self.repo, repo_type="dataset", private=False, exist_ok=True)

    def load(self) -> None:
        """Load ``state.json`` from the dataset repo into :attr:`nodes`."""
        try:
            path = hf_hub_download(self.repo, STATE_FILE, repo_type="dataset", token=self.token)
            with open(path) as f:
                raw = json.load(f)
        except (EntryNotFoundError, FileNotFoundError, RepositoryNotFoundError):
            raw = {"nodes": {}}
        nodes = {}
        for mid, d in raw.get("nodes", {}).items():
            rep = Report.from_dict(d["report"]) if d.get("report") else None
            nodes[mid] = Node(
                model_id=mid,
                report=rep,
                lineage=d.get("lineage", []),
                updated_at=d.get("updated_at"),
            )
        self.nodes = nodes

    def _serialize(self) -> dict:
        return {
            "nodes": {
                mid: {
                    "report": n.report.to_dict() if n.report else None,
                    "lineage": n.lineage,
                    "updated_at": n.updated_at,
                }
                for mid, n in self.nodes.items()
            }
        }

    def save(self) -> None:
        """Commit the in-memory state back to the dataset repo."""
        msg = f"DIA state update {datetime.datetime.now(tz=UTC).isoformat()}"
        flat_path, flat_bytes = _encode_flat_export(flatten_nodes(self.nodes))
        ops = [
            CommitOperationAdd(
                path_in_repo=STATE_FILE,
                path_or_fileobj=json.dumps(self._serialize(), indent=2).encode(),
            ),
            CommitOperationAdd(path_in_repo=flat_path, path_or_fileobj=flat_bytes),
        ]
        self.api.create_commit(
            repo_id=self.repo,
            repo_type="dataset",
            operations=ops,
            commit_message=msg,
        )

    # ---- mutations -----------------------------------------------------------
    def upsert(self, node: Node, persist: bool = True) -> None:
        """Insert or replace a single node.

        Parameters
        ----------
        node : Node
            The node to store.
        persist : bool, optional
            Commit to the dataset repo immediately. Defaults to ``True``.
        """
        with self._lock:
            node.updated_at = datetime.datetime.now(tz=UTC).isoformat()
            self.nodes[node.model_id] = node
            if persist:
                self.save()

    def upsert_many(self, nodes: list[Node]) -> None:
        """Insert or replace several nodes and commit once.

        Parameters
        ----------
        nodes : list of Node
            The nodes to store.
        """
        with self._lock:
            ts = datetime.datetime.now(tz=UTC).isoformat()
            for n in nodes:
                n.updated_at = ts
                self.nodes[n.model_id] = n
            self.save()
