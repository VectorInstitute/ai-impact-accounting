"""Self-hostable web dashboard and Hugging Face webhook Space (extra: dashboard)."""

from .server import create_app, register_ingest_webhook, serve


__all__ = ["create_app", "register_ingest_webhook", "serve"]
