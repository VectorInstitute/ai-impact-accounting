#!/usr/bin/env python3
"""Ingest DIA model repos into a HF dataset (state.json + nodes.parquet).

Usage:
    export DIA_DATASET=DIA-MVP/dia-state-lab-2026
    export HF_TOKEN=...   # or hf auth login
    python scripts/ingest_all.py

Fresh restart (empty table, then ingest):
    python scripts/ingest_all.py --reset

See LAB.md for the full A100 / A40 / CPU training plan.
"""

from __future__ import annotations

import argparse
import os
import sys

from ai_impact_accounting import Store, ingest_model

DEFAULT_DATASET = "DIA-MVP/dia-state-lab-2026"

# Uniform A100 batch — train these repos first (see LAB.md), then ingest.
MODELS_A100 = [
    "DIA-MVP/my-bert-sentiment-a100",
    "DIA-MVP/llama32-3b-lora-a100",
    "DIA-MVP/qwen2.5-7b-lora-a100",
    "DIA-MVP/resnet50-cifar100-a100",
    "DIA-MVP/cifar10-simclr-a100",
    "DIA-MVP/mnist-ddpm-a100",
    "DIA-MVP/tinyllama-lora-a100",
]

# Add after A40 runs (append to MODELS or run ingest again).
MODELS_A40 = [
    "DIA-MVP/my-bert-sentiment-a40",
    "DIA-MVP/tinyllama-lora-a40",
    "DIA-MVP/cifar10-simclr-a40",
]

# Add after CPU runs.
MODELS_CPU = [
    "DIA-MVP/my-bert-sentiment-cpu",
    "DIA-MVP/tinyllama-lora-cpu",
    "DIA-MVP/mnist-ddpm-cpu",
]

# Default ingest list for the 2026 lab restart (A100 only until you train A40/CPU).
MODELS = MODELS_A100


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest DIA model cards into a HF dataset.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the dataset store before ingesting (fresh restart).",
    )
    parser.add_argument(
        "--hardware",
        choices=("a100", "a40", "cpu", "all"),
        default="a100",
        help="Which model list to ingest (default: a100).",
    )
    args = parser.parse_args()

    if args.hardware == "a100":
        models = MODELS_A100
    elif args.hardware == "a40":
        models = MODELS_A40
    elif args.hardware == "cpu":
        models = MODELS_CPU
    else:
        models = MODELS_A100 + MODELS_A40 + MODELS_CPU

    dataset = os.environ.get("DIA_DATASET", DEFAULT_DATASET)
    store = Store(dataset)
    if args.reset:
        store.nodes.clear()
        print(f"Reset: cleared in-memory store for {dataset}")

    ok, skip = 0, 0
    for mid in models:
        try:
            r = ingest_model(mid, store, persist=False)
            if not r.get("ok"):
                print(f"  SKIP {mid} ({r.get('error', 'fetch failed')})")
                skip += 1
                continue
            if r.get("has_report"):
                print(f"  OK   {mid}")
            else:
                print(f"  ---- {mid} (repo exists, no dia_report — train + t.push() first)")
            ok += 1
        except Exception as e:
            print(f"  SKIP {mid} ({e})")
            skip += 1

    store.save()
    with_report = sum(1 for n in store.nodes.values() if n.has_report)
    print(f"\nDataset: https://huggingface.co/datasets/{dataset}")
    print(f"Nodes: {len(store.nodes)} ({with_report} with dia_report)  |  ok: {ok}  |  skipped: {skip}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
