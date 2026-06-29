#!/usr/bin/env python3
"""Remove lab model repos + dataset, then you retrain and ingest fresh.

Deletes ONLY the 2026 lab assets by default:
  - Model repos: MODELS_A100 from ingest_all.py
  - Dataset: DIA-MVP/dia-state-lab-2026

Does NOT touch public demos (my-bert-sentiment, dia-state, etc.) unless --include-legacy.

Usage:
    python scripts/reset_lab.py --dry-run     # show what would be deleted
    python scripts/reset_lab.py --yes         # actually delete

Then retrain (LAB.md) and:
    python scripts/ingest_all.py --reset
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_all import DEFAULT_DATASET, MODELS_A100  # noqa: E402

LOCAL_OUT_DIRS = [
    "out-bert",
    "out-llama-lora",
    "out-qwen-lora",
    "out-resnet50",
    "out-simclr",
    "out-ddpm",
]

# Original DIA-MVP demos — only deleted with --include-legacy (destructive).
LEGACY_MODELS = [
    "DIA-MVP/my-bert-sentiment",
    "DIA-MVP/bert-tiny-sst2-distill",
    "DIA-MVP/tinyllama-lora-demo",
    "DIA-MVP/llama32-3b-lora",
    "DIA-MVP/qwen2.5-7b-lora-demo",
    "DIA-MVP/resnet50-cifar100",
    "DIA-MVP/cifar10-simclr-resnet18",
    "DIA-MVP/mnist-ddpm",
]


def _delete_repo(api: HfApi, repo_id: str, repo_type: str, dry_run: bool) -> str:
    try:
        api.repo_info(repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        return "missing"
    except HfHubHTTPError as e:
        return f"error ({e})"
    if dry_run:
        return "would delete"
    try:
        api.delete_repo(repo_id, repo_type=repo_type)
        return "deleted"
    except HfHubHTTPError as e:
        return f"error ({e})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Wipe lab HF repos + dataset for a fresh restart.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without deleting.")
    parser.add_argument("--yes", action="store_true", help="Required to actually delete.")
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also delete original DIA-MVP demo model repos (destructive).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Also remove local out-* checkpoint folders.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Lab dataset to delete (default: {DEFAULT_DATASET}).",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.yes:
        print("Refusing to delete without --yes. Use --dry-run to preview.", file=sys.stderr)
        return 1

    dry = args.dry_run
    api = HfApi()

    models = list(MODELS_A100)
    if args.include_legacy:
        models.extend(LEGACY_MODELS)

    print("=== Hugging Face model repos ===")
    for mid in models:
        status = _delete_repo(api, mid, "model", dry)
        print(f"  {mid}: {status}")

    print(f"\n=== Dataset {args.dataset} ===")
    status = _delete_repo(api, args.dataset, "dataset", dry)
    print(f"  {status}")

    if args.local:
        print("\n=== Local checkpoint dirs ===")
        for name in LOCAL_OUT_DIRS:
            path = ROOT / name
            if not path.exists():
                print(f"  {name}: missing")
                continue
            if dry:
                print(f"  {name}: would delete")
            else:
                shutil.rmtree(path)
                print(f"  {name}: deleted")

    print()
    if dry:
        print("Dry run only. Re-run with --yes to delete.")
    else:
        print("Done. Next:")
        print("  1. Train all models (LAB.md A100 batch)")
        print("  2. python scripts/ingest_all.py --reset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
