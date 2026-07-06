"""Public, read-only DIA footprint dashboard (Hugging Face Space entrypoint)."""

import os

from ai_impact_accounting.dashboard.server import create_app
from ai_impact_accounting.hub.store import Store


os.environ.setdefault("DIA_DATASET", "DIA-MVP/dia-state-lab-2026")
DATASET = os.environ["DIA_DATASET"]
DEFAULT_BASE = os.environ.get("DIA_BASES", "distilbert-base-uncased").split(",")[0].strip()

store = Store(DATASET)
app = create_app(store, default_base=DEFAULT_BASE)
