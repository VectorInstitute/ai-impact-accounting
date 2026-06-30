"""Local dashboard viewer -- launches ONLY the Gradio UI (no webhook server).

The full Space app (``ai_impact_accounting.dashboard.app``) wraps the UI in
huggingface_hub's ``WebhooksServer``, whose routing changed between hub versions.
For local viewing the webhook server isn't needed, so this launches the dashboard
directly. Requires the ``dashboard`` extra:

    pip install "ai-impact-accounting[dashboard]"
"""

import os
import sys

import gradio as gr
from huggingface_hub import get_token

from ai_impact_accounting import Store
from ai_impact_accounting.dashboard import build_ui


DATASET = os.environ.get("DIA_DATASET", "DIA-MVP/dia-state")
DEFAULT_BASE = os.environ.get("DIA_BASES", "distilbert-base-uncased").split(",")[0].strip()
PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))


def _force_free_port(port: int) -> None:
    """Kill any process still bound to ``port`` so each launch is a clean restart.

    Re-running ``view_local.py`` after editing the UI normally fails or serves
    stale code if an old Gradio server is still holding the port. We find and
    terminate that listener first. Set ``DIA_NO_KILL=1`` to skip.
    """
    if os.environ.get("DIA_NO_KILL", "").lower() in ("1", "true", "yes"):
        return
    try:
        import psutil  # noqa: PLC0415
    except ImportError:
        return
    me = os.getpid()
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr and conn.laddr.port == port and conn.pid and conn.pid != me:
            try:
                proc = psutil.Process(conn.pid)
                proc.terminate()
                proc.wait(timeout=5)
                print(f"Force-restart: killed stale server (pid {conn.pid}) on port {port}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass


def main() -> None:
    """Load accounting state and launch the local Gradio dashboard."""
    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        print("Run: hf auth login   (or export HF_TOKEN=...)")
        sys.exit(1)
    _force_free_port(PORT)
    store = Store(DATASET, token=token)
    print(f"Loaded {len(store.nodes)} node(s) from {store.repo}")
    # share=True prints a temporary public https://*.gradio.live link (~72h) so
    # you can show the dashboard to others. Set DIA_SHARE=0 to keep it local-only.
    share = os.environ.get("DIA_SHARE", "1") != "0"
    build_ui(store, default_base=DEFAULT_BASE).launch(
        share=share, server_port=PORT, theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
