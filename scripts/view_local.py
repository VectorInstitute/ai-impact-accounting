"""Local dashboard viewer -- launches ONLY the Gradio UI (no webhook server).

The full Space app (``ai_impact_accounting.dashboard.app``) wraps the UI in
huggingface_hub's ``WebhooksServer``, whose routing changed between hub versions.
For local viewing the webhook server isn't needed, so this launches the dashboard
directly. Requires the ``dashboard`` extra:

    pip install "ai-impact-accounting[dashboard]"
"""

import os
import sys

from huggingface_hub import HfFolder

from ai_impact_accounting import Store
from ai_impact_accounting.dashboard import build_ui


DATASET = os.environ.get("DIA_DATASET", "DIA-MVP/dia-state")
DEFAULT_BASE = os.environ.get("DIA_BASES", "distilbert-base-uncased").split(",")[0].strip()


def main() -> None:
    """Load accounting state and launch the local Gradio dashboard."""
    token = os.getenv("HF_TOKEN") or HfFolder.get_token()
    if not token:
        print("Run: huggingface-cli login   (or export HF_TOKEN=...)")
        sys.exit(1)
    store = Store(DATASET, token=token)
    print(f"Loaded {len(store.nodes)} node(s) from {store.repo}")
    # share=True prints a temporary public https://*.gradio.live link (~72h) so
    # you can show the dashboard to others. Set DIA_SHARE=0 to keep it local-only.
    share = os.environ.get("DIA_SHARE", "1") != "0"
    build_ui(store, default_base=DEFAULT_BASE).launch(share=share)


if __name__ == "__main__":
    main()
