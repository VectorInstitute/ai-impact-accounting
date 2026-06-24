"""Ingest one model: fetch its card, parse the DIA block, upsert into the store.

Shared by the webhook receiver (push path) and the crawler (pull path).
"""

from __future__ import annotations

import os
from typing import Optional

from huggingface_hub import ModelCard

from ..models import Node
from ..parse import parse_lineage, parse_report
from .store import Store


def fetch_meta(model_id: str, token: Optional[str] = None) -> dict:
    """Return a model card's front-matter metadata as a plain dict.

    Parameters
    ----------
    model_id : str
        Hub model repo id.
    token : str, optional
        Hugging Face token; falls back to ``HF_TOKEN``.

    Returns
    -------
    dict
        The card metadata.
    """
    card = ModelCard.load(model_id, token=token or os.getenv("HF_TOKEN"))
    return card.data.to_dict() if hasattr(card.data, "to_dict") else dict(card.data)


def ingest_model(
    model_id: str,
    store: Store,
    token: Optional[str] = None,
    persist: bool = True,
) -> dict:
    """Fetch, parse, and store one model.

    Parameters
    ----------
    model_id : str
        Hub model repo id.
    store : Store
        The accounting store to upsert into.
    token : str, optional
        Hugging Face token; falls back to ``HF_TOKEN``.
    persist : bool, optional
        Commit immediately. Defaults to ``True``.

    Returns
    -------
    dict
        Outcome with ``ok``, ``has_report``, ``lineage``, and any ``errors``.
    """
    try:
        meta = fetch_meta(model_id, token)
    except Exception as exc:
        # Do not propagate raw Hub/API messages — they may echo request context.
        return {"model": model_id, "ok": False, "error": type(exc).__name__}

    report, errors = parse_report(meta)
    lineage = parse_lineage(meta)
    node = Node(model_id=model_id, report=report, lineage=lineage)
    store.upsert(node, persist=persist)
    return {
        "model": model_id,
        "ok": True,
        "has_report": report is not None,
        "lineage": [edge["model"] for edge in lineage],
        "errors": errors,
    }
