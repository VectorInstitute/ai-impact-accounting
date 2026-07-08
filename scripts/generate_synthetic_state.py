#!/usr/bin/env python3
"""Generate a synthetic DIA ``state.json`` for dashboard stress testing.

Creates a branching lineage tree with fake incremental footprints — no Hub access
required. Use with :class:`LocalStore` via ``DIA_STATE_FILE``.

Example::

    python scripts/generate_synthetic_state.py --nodes 100
    DIA_STATE_FILE=tests/fixtures/synth-100-state.json \\
    DIA_BASES=SYNTH-LAB/base-model \\
    python scripts/view_local.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_impact_accounting.models import Interval, Node, Report  # noqa: E402

ORG = "SYNTH-LAB"
DEFAULT_BASE = f"{ORG}/base-model"
RELATIONS = ("finetune", "lora", "adapter", "quantized", "merge", "distill")
GPUS = ("A100", "A40", "T4", "H100", "L40S")
REGIONS = ("us-east-1", "eu-west-1", "ca-central-1", "ap-south-1")
QUALITIES = ("measured", "estimated", "estimated (hardware)", "imputed")


def _make_report(
    rng: random.Random,
    *,
    carbon: float,
    method: str,
    with_report: bool,
) -> Report | None:
    if not with_report:
        return None
    quality = rng.choices(
        QUALITIES,
        weights=(62, 18, 12, 8),
        k=1,
    )[0]
    rep = Report(scope="incremental")
    jitter = rng.uniform(0.85, 1.15)
    c = carbon * jitter
    rep.carbon = Interval(c * 0.95, c * 1.05)
    rep.energy = Interval(c * 1.8, c * 2.2)
    rep.water = Interval(c * 40, c * 60)
    rep.quality = {"carbon": quality, "energy": quality, "water": quality}
    rep.method = method
    rep.gpu = rng.choice(GPUS)
    rep.gpu_count = rng.randint(1, 8)
    rep.gpu_hours = round(rng.uniform(0.5, 120.0), 2)
    rep.region = rng.choice(REGIONS)
    rep.tool = "synthetic-generator"
    return rep


def build_synthetic_nodes(
    n_nodes: int,
    *,
    org: str = ORG,
    seed: int = 42,
    report_rate: float = 0.78,
) -> dict[str, Node]:
    """Build a single-tree synthetic lineage with ``n_nodes`` models."""
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")

    rng = random.Random(seed)
    base_id = f"{org}/base-model"
    nodes: dict[str, Node] = {}
    now = datetime.now(tz=UTC).isoformat()

    nodes[base_id] = Node(
        model_id=base_id,
        report=_make_report(rng, carbon=12.0, method="finetune", with_report=True),
        lineage=[],
        updated_at=now,
    )

    frontier = [base_id]
    idx = 0
    while len(nodes) < n_nodes and frontier:
        parent = frontier.pop(0)
        # Wider fan-out near the root, narrower deeper in the tree.
        depth = parent.count("/") + parent.count("-")
        fanout = rng.randint(2, 6) if len(nodes) < n_nodes * 0.35 else rng.randint(1, 4)
        for _ in range(fanout):
            if len(nodes) >= n_nodes:
                break
            child_id = f"{org}/model-{idx:03d}"
            idx += 1
            relation = rng.choice(RELATIONS)
            with_report = rng.random() < report_rate
            carbon = rng.uniform(0.002, 2.5) * (0.35 if relation in ("lora", "adapter", "quantized") else 1.0)
            nodes[child_id] = Node(
                model_id=child_id,
                report=_make_report(rng, carbon=carbon, method=relation, with_report=with_report),
                lineage=[{"model": parent, "relation": relation}],
                updated_at=now,
            )
            frontier.append(child_id)

    return nodes


def serialize_nodes(nodes: dict[str, Node]) -> dict:
    return {
        "nodes": {
            mid: {
                "report": n.report.to_dict() if n.report else None,
                "lineage": n.lineage,
                "updated_at": n.updated_at,
            }
            for mid, n in nodes.items()
        }
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic DIA state.json for stress tests.")
    ap.add_argument("--nodes", type=int, default=100, help="Target number of models (default: 100)")
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "synth-100-state.json",
        help="Output path for state.json",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-rate", type=float, default=0.78, help="Fraction of models with dia_report")
    ap.add_argument("--org", default=ORG, help="Synthetic org prefix (default: SYNTH-LAB)")
    args = ap.parse_args()

    nodes = build_synthetic_nodes(
        args.nodes,
        org=args.org,
        seed=args.seed,
        report_rate=args.report_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_nodes(nodes)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    n_report = sum(1 for n in nodes.values() if n.report)
    print(f"Wrote {len(nodes)} node(s) ({n_report} with reports) → {args.output}")
    print()
    print("Run the dashboard against this file:")
    print(f"  DIA_STATE_FILE={args.output} \\")
    print(f"  DIA_BASES={args.org}/base-model \\")
    print("  python scripts/view_local.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
