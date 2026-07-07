"""Pull path: discover third-party derivatives by periodic crawling.

Webhooks only fire for repos you own/watch, so other people's derivatives are
found by crawling. For each tracked base model, list models that declare it as
``base_model`` and ingest each. Run nightly: webhooks keep your own corner live in
real time; this fills in everyone else's.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from collections.abc import Callable
from typing import Optional

from huggingface_hub import HfApi

from .ingest import ingest_model
from .store import Store


def list_derivatives(base: str, token: Optional[str] = None, limit: int = 1000) -> list[str]:
    """List Hub models that declare ``base`` as their base model.

    Parameters
    ----------
    base : str
        Base model repo id.
    token : str, optional
        Hugging Face token; falls back to ``HF_TOKEN``.
    limit : int, optional
        Maximum models to list per filter. Defaults to ``1000``.

    Returns
    -------
    list of str
        Sorted derivative model ids (excluding the base itself).
    """
    api = HfApi(token=token or os.getenv("HF_TOKEN"))
    ids = set()
    # HF tags derivatives as base_model:<relation>:<base>; filtering on the
    # base_model facet returns the family. Fall back to a search if needed.
    for flt in (f"base_model:{base}", base):
        try:
            for m in api.list_models(filter=flt, limit=limit):
                ids.add(m.id)
        except Exception:
            try:
                for m in api.list_models(search=base, limit=limit):
                    ids.add(m.id)
            except Exception:
                pass
    ids.discard(base)
    return sorted(ids)


def crawl_once(store: Store, bases: list[str], token: Optional[str] = None) -> dict:
    """Crawl every tracked base and its derivatives, committing once.

    Parameters
    ----------
    store : Store
        The accounting store to populate.
    bases : list of str
        Base model ids to expand.
    token : str, optional
        Hugging Face token; falls back to ``HF_TOKEN``.

    Returns
    -------
    dict
        ``{"crawled": <count>}``.
    """
    token = token or os.getenv("HF_TOKEN")
    seen = set(bases)
    for base in bases:
        ingest_model(base, store, token=token, persist=False)
        for mid in list_derivatives(base, token=token):
            if mid not in seen:
                ingest_model(mid, store, token=token, persist=False)
                seen.add(mid)
    store.save()  # one commit for the whole crawl
    return {"crawled": len(seen)}


def crawl_orgs(
    store: Store,
    orgs: list[str],
    token: Optional[str] = None,
    require_report: bool = True,
) -> dict:
    """Ingest every model under each org/user, committing once.

    Discovery by owner rather than by base: whatever the team publishes is
    indexed, regardless of its lineage or whether it derives from a tracked
    base. The org list is small and stable; a base list never is.

    Parameters
    ----------
    store : Store
        The accounting store to populate.
    orgs : list of str
        Hugging Face org or user names to list models for.
    token : str, optional
        Hugging Face token; falls back to ``HF_TOKEN``.
    require_report : bool, optional
        Keep only models that disclose a ``dia_report``. Defaults to ``True``.

    Returns
    -------
    dict
        ``{"crawled": <listed>, "kept": <stored>}``.
    """
    token = token or os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    seen: set[str] = set()
    for org in orgs:
        for m in api.list_models(author=org):
            if m.id in seen:
                continue
            seen.add(m.id)
            res = ingest_model(m.id, store, token=token, persist=False)
            if require_report and res.get("ok") and not res.get("has_report"):
                store.nodes.pop(m.id, None)
    store.save()  # one commit for the whole crawl
    return {"crawled": len(seen), "kept": len(store.nodes)}


def start_scheduler(
    store: Store,
    bases_getter: Callable[[], list[str]],
    token: Optional[str] = None,
    interval_s: int = 24 * 3600,
) -> threading.Thread:
    """Start a daemon thread that re-crawls on a fixed interval.

    Parameters
    ----------
    store : Store
        The accounting store to populate.
    bases_getter : callable
        Zero-argument callable returning the current list of base ids.
    token : str, optional
        Hugging Face token; falls back to ``HF_TOKEN``.
    interval_s : int, optional
        Seconds between crawls. Defaults to one day.

    Returns
    -------
    threading.Thread
        The started daemon thread.
    """

    def _loop() -> None:
        while True:
            try:
                crawl_once(store, bases_getter(), token=token)
            except Exception:
                traceback.print_exc()
            time.sleep(interval_s)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t
