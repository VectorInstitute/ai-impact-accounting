#!/usr/bin/env python3
"""Ingest local model cards into a HF dataset — no Hub model repo required.

Use when training wrote ``dia_report`` to ``out-*/README.md`` locally but Hub
push failed or was skipped (``DIA_LOCAL=1``). Metadata lands in the dataset
``state.json``; weights stay on disk.

Usage::

    export DIA_DATASET=vector-institute/dia-state-lab-2026
    export HF_TOKEN=...   # write access to the *dataset* repo

    # model_id=path/to/README.md (repeatable)
    python scripts/ingest_local_cards.py \\
      DIA-MVP/my-bert-sentiment-a40-v3=batch_jobs/local_runs/bert-a40-v3/README.md \\
      DIA-MVP/my-bert-sentiment-euwest=batch_jobs/local_runs/bert-euwest/README.md

    # scan a directory of per-run folders (each contains README.md)
    python scripts/ingest_local_cards.py --runs-dir batch_jobs/local_runs

``--runs-dir`` expects ``<runs-dir>/<model_id_with_slashes_as_dirs>/README.md``
or ``<runs-dir>/<slug>/README.md`` with a sidecar ``model_id`` file, OR folders
named by the last segment with a ``mapping.json`` — simplest is explicit pairs.

For enrichment batch jobs, give each run a unique ``OUT``::

    DIA_LOCAL=1 OUT=batch_jobs/local_runs/bert-a40-v3 REPO=DIA-MVP/my-bert-sentiment-a40-v3 \\
      python scripts/train_bert_demo.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ai_impact_accounting import Store
from ai_impact_accounting.hub.ingest import fetch_model_card
from ai_impact_accounting.models import Node
from ai_impact_accounting.parse import parse_lineage, parse_report

DEFAULT_DATASET = "vector-institute/dia-state-lab-2026"


def ingest_local_card(
    model_id: str,
    card_path: Path,
    store: Store,
    *,
    persist: bool = False,
) -> dict:
    """Parse a local README and upsert under ``model_id``."""
    if not card_path.is_file():
        return {"model": model_id, "ok": False, "error": f"card not found: {card_path}"}

    try:
        meta, _ = fetch_model_card(str(card_path))
    except Exception as exc:  # noqa: BLE001
        return {"model": model_id, "ok": False, "error": type(exc).__name__}

    report, errors = parse_report(meta)
    lineage = parse_lineage(meta)
    node = Node(model_id=model_id, report=report, lineage=lineage)
    store.upsert(node, persist=persist)
    return {
        "model": model_id,
        "ok": True,
        "has_report": report is not None,
        "lineage": [e["model"] for e in lineage],
        "errors": errors,
    }


def _pairs_from_runs_dir(runs_dir: Path) -> list[tuple[str, Path]]:
    """Load ``mapping.json`` or infer one README per subfolder."""
    mapping_file = runs_dir / "mapping.json"
    if mapping_file.is_file():
        raw = json.loads(mapping_file.read_text(encoding="utf-8"))
        return [(mid, runs_dir / rel / "README.md" if not str(rel).endswith(".md") else runs_dir / rel)
                for mid, rel in raw.items()]

    pairs: list[tuple[str, Path]] = []
    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        card = sub / "README.md"
        model_id_file = sub / "model_id"
        if card.is_file() and model_id_file.is_file():
            pairs.append((model_id_file.read_text(encoding="utf-8").strip(), card))
    return pairs


def _parse_pair(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError(f"expected model_id=path, got {arg!r}")
    mid, path = arg.split("=", 1)
    return mid.strip(), Path(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest local dia_report cards into a HF dataset.")
    ap.add_argument(
        "pairs",
        nargs="*",
        type=_parse_pair,
        help="model_id=path/to/README.md",
    )
    ap.add_argument(
        "--runs-dir",
        type=Path,
        help="Directory of per-run subfolders (see mapping.json or model_id files).",
    )
    ap.add_argument(
        "--dataset",
        default=os.environ.get("DIA_DATASET", DEFAULT_DATASET),
        help=f"Target dataset (default: {DEFAULT_DATASET})",
    )
    ap.add_argument("--dry-run", action="store_true", help="Parse cards only; do not write dataset.")
    args = ap.parse_args()

    pairs: list[tuple[str, Path]] = list(args.pairs)
    if args.runs_dir:
        pairs.extend(_pairs_from_runs_dir(args.runs_dir))

    if not pairs:
        ap.error("provide model_id=path pairs and/or --runs-dir")

    ok = skip = 0

    if args.dry_run:
        for model_id, card_path in pairs:
            if not card_path.is_file():
                print(f"  SKIP {model_id} (missing {card_path})")
                skip += 1
                continue
            meta, _ = fetch_model_card(str(card_path))
            report, errs = parse_report(meta)
            if report:
                print(f"  OK   {model_id}  ←  {card_path}")
                ok += 1
            else:
                print(f"  ---- {model_id}  (no dia_report)  errs={errs}")
                skip += 1
        return 0 if ok else 1

    store = Store(args.dataset)
    for model_id, card_path in pairs:
        r = ingest_local_card(model_id, card_path, store, persist=False)
        if not r.get("ok"):
            print(f"  SKIP {model_id} ({r.get('error')})")
            skip += 1
        elif r.get("has_report"):
            print(f"  OK   {model_id}  ←  {card_path}")
            ok += 1
        else:
            print(f"  ---- {model_id}  (no usable dia_report)")
            skip += 1

    store.save()
    with_report = sum(1 for n in store.nodes.values() if n.has_report)
    print(f"\nDataset: https://huggingface.co/datasets/{args.dataset}")
    print(f"Nodes: {len(store.nodes)} ({with_report} with dia_report)  |  ok: {ok}  |  skipped: {skip}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
