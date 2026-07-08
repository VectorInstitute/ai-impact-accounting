"""Public DIA footprint dashboard (Hugging Face Space entrypoint).

Read-only by default: serves the dashboard against a public dataset, no token
needed. If the Space is configured with write credentials, it also exposes a
real-time ingest webhook so a model push adds/updates its node in the dataset.

Optional Space secrets (enable real-time ingest when BOTH are set):

- ``HF_TOKEN``        -- write token for the dataset repo
- ``WEBHOOK_SECRET``  -- shared secret; must match the Hub webhook config

Webhook URL (Hub convention): ``https://<space>.hf.space/webhooks/webhooks/ingest``

The periodic backfill crawler runs *outside* the Space (see scripts/crawl.py and
the nightly GitHub Actions workflow), so this process holds no long-lived thread.
"""

import os

from ai_impact_accounting.dashboard.server import create_app, register_ingest_webhook
from ai_impact_accounting.hub.store import Store


os.environ.setdefault("DIA_DATASET", "DIA-MVP/dia-state-lab-2026")
DATASET = os.environ["DIA_DATASET"]
DEFAULT_BASE = os.environ.get("DIA_BASES", "distilbert-base-uncased").split(",")[0].strip()

HF_TOKEN = os.environ.get("HF_TOKEN")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")

store = Store(DATASET, token=HF_TOKEN)
app = create_app(store, default_base=DEFAULT_BASE)


if HF_TOKEN and WEBHOOK_SECRET:
    from huggingface_hub import WebhookPayload

    from ai_impact_accounting.hub import ingest_model

    async def _ingest(payload: WebhookPayload) -> dict:
        """Ingest a model on a content-changing push to a model repo."""
        if payload.repo.type != "model":
            return {"processed": False, "reason": "not a model"}
        if payload.event.action not in ("create", "update"):
            return {"processed": False, "reason": payload.event.action}
        if not payload.event.scope.startswith("repo.content"):
            return {"processed": False, "reason": payload.event.scope}
        res = ingest_model(payload.repo.name, store, token=HF_TOKEN, persist=True)
        return {"processed": res["ok"], **res}

    register_ingest_webhook(app, _ingest, webhook_secret=WEBHOOK_SECRET)
