#!/usr/bin/env python3
"""Deploy the public, read-only DIA dashboard to a Hugging Face Space.

Vendors the (light) ``ai_impact_accounting`` package into the Space so the build
does not pull torch/transformers -- the dashboard only reads the public dataset.

The same ``space/`` files serve any target Space; the dataset the dashboard reads
is baked into the uploaded ``Dockerfile`` (``ENV DIA_DATASET=...``) at deploy time,
so one script can publish to multiple Spaces that read different datasets.

Usage:
    python scripts/deploy_space.py                       # default Space + dataset
    python scripts/deploy_space.py --status              # print runtime status
    python scripts/deploy_space.py \\
        --space vector-institute/dia-dashboard \\
        --dataset vector-institute/dia-state-lab-2026    # publish elsewhere
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_SPACE = "DIA-MVP/dia-dashboard"
DEFAULT_DATASET = "DIA-MVP/dia-state-lab-2026"

SPACE_DIR = Path(__file__).resolve().parent.parent / "space"


def _render_dockerfile(dataset: str) -> bytes:
    """Return the Space Dockerfile with ``ENV DIA_DATASET`` set to ``dataset``."""
    text = (SPACE_DIR / "Dockerfile").read_text()
    new, n = re.subn(
        r"^ENV DIA_DATASET=.*$",
        f"ENV DIA_DATASET={dataset}",
        text,
        flags=re.MULTILINE,
    )
    if n == 0:
        raise SystemExit("space/Dockerfile has no 'ENV DIA_DATASET=' line to set.")
    return new.encode()


def _render_readme(dataset: str) -> bytes:
    """Return the Space README with its dataset link pointed at ``dataset``."""
    text = (SPACE_DIR / "README.md").read_text()
    return text.replace(DEFAULT_DATASET, dataset).encode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=DEFAULT_SPACE, help="Target HF Space id.")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="DIA_DATASET the Space reads.")
    ap.add_argument("--status", action="store_true", help="Only print runtime status.")
    args = ap.parse_args()
    api = HfApi()
    space = args.space

    if args.status:
        rt = api.get_space_runtime(space)
        print(f"{space}: stage={rt.stage} hardware={rt.hardware}")
        print(f"https://huggingface.co/spaces/{space}")
        return 0

    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    # Vendor the package (skip caches).
    api.upload_folder(
        folder_path="src/ai_impact_accounting",
        path_in_repo="ai_impact_accounting",
        repo_id=space,
        repo_type="space",
        ignore_patterns=["__pycache__/*", "*.pyc"],
        commit_message="Deploy DIA dashboard package",
    )
    # ``Dockerfile`` must land at the Space root for the docker SDK to build; its
    # DIA_DATASET is rendered per-target so each Space reads the right dataset.
    api.upload_file(
        path_or_fileobj=_render_dockerfile(args.dataset),
        path_in_repo="Dockerfile",
        repo_id=space,
        repo_type="space",
        commit_message=f"Deploy Dockerfile (DIA_DATASET={args.dataset})",
    )
    # README's dataset link is rendered per-target too (frontmatter/title stay shared).
    api.upload_file(
        path_or_fileobj=_render_readme(args.dataset),
        path_in_repo="README.md",
        repo_id=space,
        repo_type="space",
        commit_message="Deploy README.md",
    )
    for f in ("app.py", "requirements.txt"):
        api.upload_file(
            path_or_fileobj=f"space/{f}",
            path_in_repo=f,
            repo_id=space,
            repo_type="space",
            commit_message=f"Deploy {f}",
        )
    print(f"Deployed: https://huggingface.co/spaces/{space} (dataset: {args.dataset})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
