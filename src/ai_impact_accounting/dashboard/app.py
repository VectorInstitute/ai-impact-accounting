"""Hugging Face Space entrypoint (dashboard + webhooks + crawler).

Combines:

- the DIA web dashboard (:mod:`ai_impact_accounting.dashboard.server`),
- a webhook endpoint that ingests models on push (real-time, your repos),
- a nightly crawler that backfills third-party derivatives (pull path).

Required Space secrets:

- ``HF_TOKEN`` -- write token (to commit state to the dataset repo)
- ``WEBHOOK_SECRET`` -- shared secret, must match the webhook config on the Hub
- ``DIA_DATASET`` -- e.g. ``your-username/dia-state``
- ``DIA_BASES`` -- comma-separated base models to track

Webhook URL (Hub convention): ``https://<space>.hf.space/webhooks/webhooks/ingest``
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import WebhookPayload, get_token

from ..hub import Store, ingest_model, start_scheduler
from .server import create_app, register_ingest_webhook


HF_TOKEN = os.getenv("HF_TOKEN") or get_token()
if not HF_TOKEN:
    print("Run: hf auth login   (or export HF_TOKEN=...)", file=sys.stderr)
    sys.exit(1)
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")
DATASET = os.environ.get("DIA_DATASET", "dia-state")
ENV_BASES = [b.strip() for b in os.environ.get("DIA_BASES", "meta-llama/Llama-3-8B").split(",") if b.strip()]

store = Store(DATASET, token=HF_TOKEN)
DEFAULT_BASE = ENV_BASES[0] if ENV_BASES else "meta-llama/Llama-3-8B"


def tracked_bases() -> list[str]:
    """Return env-declared bases plus any base referenced by stored lineage."""
    bases = set(ENV_BASES)
    for n in store.nodes.values():
        for parent in n.lineage:
            if parent.get("model"):
                bases.add(parent["model"])
    return sorted(bases)


app = create_app(store, default_base=DEFAULT_BASE)


async def ingest(payload: WebhookPayload) -> dict:
    """Ingest a model on a content-changing push to a model repo."""
    if payload.repo.type != "model":
        return {"processed": False, "reason": "not a model"}
    if payload.event.action not in ("create", "update"):
        return {"processed": False, "reason": payload.event.action}
    if not payload.event.scope.startswith("repo.content"):
        return {"processed": False, "reason": payload.event.scope}

    res = ingest_model(payload.repo.name, store, token=HF_TOKEN, persist=True)
    return {"processed": res["ok"], **res}


register_ingest_webhook(app, ingest, webhook_secret=WEBHOOK_SECRET)


if __name__ == "__main__":
    import uvicorn  # noqa: PLC0415

    start_scheduler(store, tracked_bases, token=HF_TOKEN, interval_s=24 * 3600)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
