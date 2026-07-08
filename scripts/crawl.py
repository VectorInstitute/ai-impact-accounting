#!/usr/bin/env python3
"""Backfill the DIA dataset by crawling the Hub, then restart the Space.

Discovery runs two complementary ways:

* ``DIA_ORGS`` (recommended): index *everything your team publishes* by listing
  every model under each org/user. The org list is small and stable, so it never
  goes stale the way a base list does — any new training shows up automatically,
  whatever base it derives from (or none at all).
* ``DIA_BASES`` (optional): also catch *third-party* derivatives of your models
  by listing everything that declares each base as its ``base_model``.

Set at least one. This is the *pull path*; run it on an external schedule (e.g.
the nightly GitHub Actions workflow), not inside the Space.

Usage:
    export DIA_DATASET=DIA-MVP/dia-state-lab-2026
    export DIA_ORGS=vector-institute,DIA-MVP           # who to index
    export DIA_BASES=distilbert-base-uncased           # optional: external forks
    export HF_TOKEN=...                                 # write token for the dataset
    export DIA_SPACE=vector-institute/dia-dashboard     # optional: restart to reload
    python scripts/crawl.py
"""

from __future__ import annotations

import os
import sys

from ai_impact_accounting import Store
from ai_impact_accounting.hub.crawl import crawl_once, crawl_orgs

DEFAULT_DATASET = "DIA-MVP/dia-state-lab-2026"


def main() -> int:
    dataset = os.environ.get("DIA_DATASET", DEFAULT_DATASET)
    orgs = [o.strip() for o in os.environ.get("DIA_ORGS", "").split(",") if o.strip()]
    bases = [b.strip() for b in os.environ.get("DIA_BASES", "").split(",") if b.strip()]
    token = os.environ.get("HF_TOKEN")

    if not orgs and not bases:
        print(
            "Nothing to crawl. Set DIA_ORGS=org1,org2 (recommended) and/or "
            "DIA_BASES=base1,base2.",
            file=sys.stderr,
        )
        return 2
    if not token:
        print("No HF_TOKEN set — a write token is required to commit the crawl.", file=sys.stderr)
        return 2

    store = Store(dataset, token=token)

    if orgs:
        print(f"Crawling {len(orgs)} org(s) into {dataset}: {', '.join(orgs)}")
        result = crawl_orgs(store, orgs, token=token)
        print(f"Listed {result['crawled']} model(s) across orgs.")
    if bases:
        print(f"Crawling {len(bases)} base(s) for external derivatives: {', '.join(bases)}")
        result = crawl_once(store, bases, token=token)
        print(f"Listed {result['crawled']} derivative model(s).")

    with_report = sum(1 for n in store.nodes.values() if n.has_report)
    print(f"Dataset: https://huggingface.co/datasets/{dataset}")
    print(f"Nodes: {len(store.nodes)} ({with_report} with dia_report)")

    # The running Space caches state.json in memory; restart it so it reloads the
    # freshly-crawled data. Requires the token to have write access to the Space.
    space = os.environ.get("DIA_SPACE")
    if space:
        from huggingface_hub import HfApi  # noqa: PLC0415

        try:
            HfApi(token=token).restart_space(space)
            print(f"Restarted Space {space} to reload the dataset.")
        except Exception as exc:
            print(f"Warning: could not restart Space {space}: {type(exc).__name__}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
