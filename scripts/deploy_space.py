#!/usr/bin/env python3
"""Deploy the public, read-only DIA dashboard to a Hugging Face Space.

Vendors the (light) ``ai_impact_accounting`` package into the Space so the build
does not pull torch/transformers -- the dashboard only reads the public dataset.

Usage:
    python scripts/deploy_space.py            # create/update the Space
    python scripts/deploy_space.py --status   # just print runtime status
"""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import HfApi

SPACE = "DIA-MVP/dia-dashboard"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="Only print runtime status.")
    args = ap.parse_args()
    api = HfApi()

    if args.status:
        rt = api.get_space_runtime(SPACE)
        print(f"{SPACE}: stage={rt.stage} hardware={rt.hardware}")
        print(f"https://huggingface.co/spaces/{SPACE}")
        return 0

    api.create_repo(SPACE, repo_type="space", space_sdk="docker", exist_ok=True)
    # Vendor the package (skip caches).
    api.upload_folder(
        folder_path="src/ai_impact_accounting",
        path_in_repo="ai_impact_accounting",
        repo_id=SPACE,
        repo_type="space",
        ignore_patterns=["__pycache__/*", "*.pyc"],
        commit_message="Deploy DIA dashboard package",
    )
    for f in ("app.py", "requirements.txt", "README.md"):
        api.upload_file(
            path_or_fileobj=f"space/{f}",
            path_in_repo=f,
            repo_id=SPACE,
            repo_type="space",
            commit_message=f"Deploy {f}",
        )
    print(f"Deployed: https://huggingface.co/spaces/{SPACE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
