"""Local dashboard viewer — Starlette/FastAPI web UI (no Gradio).

Requires the ``dashboard`` extra::

    pip install "ai-impact-accounting[dashboard]"
"""

import os
import sys

from huggingface_hub import get_token

from ai_impact_accounting import LocalStore, Store
from ai_impact_accounting.dashboard.server import serve


DATASET = os.environ.get("DIA_DATASET", "DIA-MVP/dia-state-lab-2026")
STATE_FILE = os.environ.get("DIA_STATE_FILE", "").strip()
DEFAULT_BASE = os.environ.get("DIA_BASES", "distilbert-base-uncased").split(",")[0].strip()
PORT = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))
HOST = os.environ.get("HOST", "0.0.0.0")


def _force_free_port(port: int) -> None:
    """Kill any process still bound to ``port`` so each launch is a clean restart."""
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
    """Load accounting state and launch the web dashboard."""
    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        print("No HF_TOKEN — read-only mode (public datasets only).")
    _force_free_port(PORT)
    try:
        if STATE_FILE:
            store = LocalStore(STATE_FILE)
            if DEFAULT_BASE == "distilbert-base-uncased" and store.nodes:
                synth_bases = [m for m in store.nodes if m.endswith("/base-model")]
                if synth_bases:
                    default_base = synth_bases[0]
                else:
                    default_base = next(iter(store.nodes))
            else:
                default_base = DEFAULT_BASE
        else:
            store = Store(DATASET, token=token)
            default_base = DEFAULT_BASE
    except (ValueError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(store.nodes)} node(s) from {store.repo}")
    print(f"DIA dashboard at http://127.0.0.1:{PORT}")
    serve(store, host=HOST, port=PORT, default_base=default_base)


if __name__ == "__main__":
    main()
