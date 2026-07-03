"""Self-hostable Gradio dashboard and Hugging Face webhook Space (extra: dashboard)."""

from .theme import dia_launch_kwargs, dia_theme
from .ui import build_ui


__all__ = ["build_ui", "dia_launch_kwargs", "dia_theme"]
