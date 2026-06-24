"""Hugging Face Space entrypoint.

Combines three things in one process:

- a Gradio dashboard (:mod:`ai_impact_accounting.dashboard.ui`),
- a webhook endpoint that ingests models on push (real-time, your repos),
- a nightly crawler that backfills third-party derivatives (pull path).

Required Space secrets:

- ``HF_TOKEN`` -- write token (to commit state to the dataset repo)
- ``WEBHOOK_SECRET`` -- shared secret, must match the webhook config on the Hub
- ``DIA_DATASET`` -- e.g. ``your-username/dia-state``
- ``DIA_BASES`` -- comma-separated base models to track, e.g. ``meta-llama/Llama-3-8B``

Note: the webhook route is mounted under ``/webhooks`` by ``WebhooksServer``, so the
external URL is doubled: ``https://<space>.hf.space/webhooks/webhooks/ingest``. This
module targets the pinned ``huggingface_hub`` from the ``dashboard`` extra; newer
releases change the ``WebhooksServer`` routing.
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import HfFolder, WebhookPayload, WebhooksServer

from ..hub import Store, ingest_model, start_scheduler
from .ui import build_ui


HF_TOKEN = os.getenv("HF_TOKEN") or HfFolder.get_token()
if not HF_TOKEN:
    print("Run: huggingface-cli login   (or export HF_TOKEN=...)", file=sys.stderr)
    sys.exit(1)
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")
DATASET = os.environ.get("DIA_DATASET", "dia-state")
ENV_BASES = [b.strip() for b in os.environ.get("DIA_BASES", "meta-llama/Llama-3-8B").split(",") if b.strip()]

store = Store(DATASET, token=HF_TOKEN)


def tracked_bases() -> list[str]:
    """Return env-declared bases plus any base referenced by stored lineage.

    Returns
    -------
    list of str
        Sorted unique base model ids.
    """
    bases = set(ENV_BASES)
    for n in store.nodes.values():
        for parent in n.lineage:
            if parent.get("model"):
                bases.add(parent["model"])
    return sorted(bases)


ui = build_ui(store, default_base=ENV_BASES[0] if ENV_BASES else "meta-llama/Llama-3-8B")
app = WebhooksServer(ui=ui, webhook_secret=WEBHOOK_SECRET)


@app.add_webhook("/webhooks/ingest")  # type: ignore[untyped-decorator]
async def ingest(payload: WebhookPayload) -> dict:
    """Ingest a model on a content-changing push to a model repo.

    Parameters
    ----------
    payload : WebhookPayload
        The Hugging Face webhook payload.

    Returns
    -------
    dict
        Outcome with ``processed`` plus the ingest result, or a skip ``reason``.
    """
    # Only real commits on model repos.
    if payload.repo.type != "model":
        return {"processed": False, "reason": "not a model"}
    if payload.event.action not in ("create", "update"):
        return {"processed": False, "reason": payload.event.action}
    if not payload.event.scope.startswith("repo.content"):
        return {"processed": False, "reason": payload.event.scope}

    res = ingest_model(payload.repo.name, store, token=HF_TOKEN, persist=True)
    return {"processed": res["ok"], **res}


if __name__ == "__main__":
    # nightly backfill of derivatives we don't get webhooks for
    start_scheduler(store, tracked_bases, token=HF_TOKEN, interval_s=24 * 3600)
    app.launch()
