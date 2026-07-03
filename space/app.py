"""Public, read-only DIA footprint dashboard (Hugging Face Space entrypoint).

Reads the public rollup dataset and serves only the Gradio UI -- no webhook
server and no write token required, so anyone can view it.
"""

import os

from ai_impact_accounting.dashboard.theme import dia_launch_kwargs
from ai_impact_accounting.dashboard.ui import build_ui
from ai_impact_accounting.hub.store import Store


os.environ.setdefault("DIA_DATASET", "DIA-MVP/dia-state-lab-2026")
DATASET = os.environ["DIA_DATASET"]
DEFAULT_BASE = os.environ.get("DIA_BASES", "distilbert-base-uncased").split(",")[0].strip()

store = Store(DATASET)  # public dataset -> read-only, token optional
demo = build_ui(store, default_base=DEFAULT_BASE)
demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, **dia_launch_kwargs())
