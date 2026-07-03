"""Shared Gradio theme aligned with the project site (docs/style.css)."""

from __future__ import annotations

import gradio as gr


# Light palette (docs/style.css)
L_BG = "#faf9f7"
L_SURFACE = "#ffffff"
L_SURFACE_ALT = "#f5f3ef"
L_INK = "#1a1a1a"
L_MUTED = "#5c5c5c"
L_MUTED_LIGHT = "#8a8a8a"
L_BORDER = "#e3e0da"
L_BORDER_SOFT = "#ede9e3"
L_ACCENT = "#1a5c35"
L_ACCENT_MID = "#2e7d52"
L_ACCENT_LIGHT = "#e8f3ec"
L_HEADING = "#3d3d3d"

# Dark palette (same accent family, tuned for contrast)
D_BG = "#0f1410"
D_SURFACE = "#1a201c"
D_SURFACE_ALT = "#232b26"
D_INK = "#e8ebe9"
D_MUTED = "#9aa39c"
D_MUTED_LIGHT = "#6d7770"
D_BORDER = "#2e3832"
D_BORDER_SOFT = "#343f38"
D_ACCENT = "#5cb87a"
D_ACCENT_MID = "#3d9960"
D_ACCENT_LIGHT = "#1e3328"
D_HEADING = "#c8d0cb"

# Shared component tokens — one ``.set()`` call (Gradio resets dark tokens on re-set).
_THEME_TOKENS: dict[str, str] = {
    # Body
    "body_background_fill": L_BG,
    "body_background_fill_dark": D_BG,
    "body_text_color": L_INK,
    "body_text_color_dark": D_INK,
    "body_text_color_subdued": L_MUTED,
    "body_text_color_subdued_dark": D_MUTED,
    # Surfaces
    "background_fill_primary": L_SURFACE,
    "background_fill_primary_dark": D_SURFACE,
    "background_fill_secondary": L_SURFACE_ALT,
    "background_fill_secondary_dark": D_SURFACE_ALT,
    "block_background_fill": L_SURFACE,
    "block_background_fill_dark": D_SURFACE,
    "panel_background_fill": L_SURFACE,
    "panel_background_fill_dark": D_SURFACE,
    # Borders & labels
    "block_border_color": L_BORDER,
    "block_border_color_dark": D_BORDER,
    "border_color_primary": L_BORDER,
    "border_color_primary_dark": D_BORDER,
    "block_label_text_color": L_ACCENT,
    "block_label_text_color_dark": D_ACCENT,
    "block_label_background_fill": L_ACCENT_LIGHT,
    "block_label_background_fill_dark": D_ACCENT_LIGHT,
    "block_title_text_color": L_INK,
    "block_title_text_color_dark": D_INK,
    "block_info_text_color": L_MUTED,
    "block_info_text_color_dark": D_MUTED,
    # Inputs
    "input_background_fill": L_SURFACE,
    "input_background_fill_dark": D_SURFACE_ALT,
    "input_border_color": L_BORDER,
    "input_border_color_dark": D_BORDER,
    # Buttons
    "button_primary_background_fill": L_ACCENT,
    "button_primary_background_fill_dark": D_ACCENT_MID,
    "button_primary_background_fill_hover": L_ACCENT_MID,
    "button_primary_background_fill_hover_dark": D_ACCENT,
    "button_primary_text_color": "white",
    "button_primary_text_color_dark": "white",
    "button_secondary_background_fill": L_SURFACE,
    "button_secondary_background_fill_dark": D_SURFACE_ALT,
    "button_secondary_background_fill_hover": L_SURFACE_ALT,
    "button_secondary_background_fill_hover_dark": D_BORDER,
    "button_secondary_text_color": L_INK,
    "button_secondary_text_color_dark": D_INK,
    "button_secondary_border_color": L_BORDER,
    "button_secondary_border_color_dark": D_BORDER,
    # Links
    "link_text_color": L_ACCENT,
    "link_text_color_dark": D_ACCENT,
    "link_text_color_hover": L_ACCENT_MID,
    "link_text_color_hover_dark": D_ACCENT_MID,
    # Controls
    "checkbox_background_color": L_SURFACE,
    "checkbox_background_color_dark": D_SURFACE_ALT,
    "checkbox_border_color": L_BORDER,
    "checkbox_border_color_dark": D_BORDER,
    # Code & tables
    "code_background_fill": L_SURFACE_ALT,
    "code_background_fill_dark": D_SURFACE_ALT,
    "table_odd_background_fill": L_SURFACE_ALT,
    "table_odd_background_fill_dark": D_SURFACE_ALT,
    "table_even_background_fill": L_SURFACE,
    "table_even_background_fill_dark": D_SURFACE,
    "table_text_color": L_INK,
    "table_text_color_dark": D_INK,
    "table_border_color": L_BORDER,
    "table_border_color_dark": D_BORDER,
    "table_row_focus": L_ACCENT_LIGHT,
    "table_row_focus_dark": D_ACCENT_LIGHT,
}


def dia_theme() -> gr.Theme:
    """Return the DIA dashboard theme with paired light and dark palettes."""
    return (
        gr.themes.Soft(
            primary_hue=gr.themes.colors.green,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("DM Sans"),
            font_mono=gr.themes.GoogleFont("DM Mono"),
        )
        .set(
            **_THEME_TOKENS,
            block_border_width="1px",
            block_title_text_weight="600",
        )
    )


DIA_CSS = """
/* Custom dashboard chrome — switches with Gradio's :root.dark */
:root {
  --dia-bg: #faf9f7;
  --dia-surface: #ffffff;
  --dia-surface-alt: #f5f3ef;
  --dia-ink: #1a1a1a;
  --dia-muted: #5c5c5c;
  --dia-muted-light: #8a8a8a;
  --dia-border: #e3e0da;
  --dia-border-soft: #ede9e3;
  --dia-accent: #1a5c35;
  --dia-accent-mid: #2e7d52;
  --dia-heading: #3d3d3d;
  --dia-q-measured: #e8f3ec;
  --dia-q-estimated: #fdf0e0;
  --dia-q-imputed: #eeeef2;
  --dia-q-none: #f5f3ef;
  --dia-q-none-text: #8a8a8a;
  --dia-shadow: rgba(0, 0, 0, 0.06);
}
:root.dark, :root .dark {
  --dia-bg: #0f1410;
  --dia-surface: #1a201c;
  --dia-surface-alt: #232b26;
  --dia-ink: #e8ebe9;
  --dia-muted: #9aa39c;
  --dia-muted-light: #6d7770;
  --dia-border: #2e3832;
  --dia-border-soft: #343f38;
  --dia-accent: #5cb87a;
  --dia-accent-mid: #3d9960;
  --dia-heading: #c8d0cb;
  --dia-q-measured: #1e3328;
  --dia-q-estimated: #3d3020;
  --dia-q-imputed: #2a2a32;
  --dia-q-none: #232b26;
  --dia-q-none-text: #6d7770;
  --dia-shadow: rgba(0, 0, 0, 0.25);
}

.dia-kpi-row { display: flex; flex-wrap: wrap; gap: 12px; margin: 8px 0 16px; }
.dia-kpi {
  flex: 1 1 140px;
  background: var(--dia-surface);
  border: 1px solid var(--dia-border);
  border-radius: 12px;
  padding: 14px 16px;
  min-width: 120px;
  box-shadow: 0 1px 3px var(--dia-shadow);
}
.dia-kpi-label {
  font-size: 0.78rem;
  color: var(--dia-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.dia-kpi-value {
  font-size: 1.35rem;
  font-weight: 600;
  color: var(--dia-ink);
  margin-top: 4px;
}
.dia-kpi-sub { font-size: 0.8rem; color: var(--dia-muted-light); margin-top: 2px; }

.dia-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.dia-table th {
  text-align: left;
  padding: 10px 12px;
  background: var(--dia-surface-alt);
  border-bottom: 2px solid var(--dia-border);
  font-weight: 600;
  color: var(--dia-heading);
}
.dia-table td {
  padding: 9px 12px;
  border-bottom: 1px solid var(--dia-border-soft);
  vertical-align: top;
  color: var(--dia-ink);
}
.dia-table tr:hover td { filter: brightness(0.97); }
.dia-table a { color: var(--dia-accent); text-decoration: none; font-weight: 500; }
.dia-table a:hover { text-decoration: underline; }

.q-measured { background: var(--dia-q-measured); }
.q-estimated { background: var(--dia-q-estimated); }
.q-imputed { background: var(--dia-q-imputed); }
.q-none { background: var(--dia-q-none); color: var(--dia-q-none-text); }

.dia-meta { color: var(--dia-muted); font-size: 0.92rem; margin-bottom: 8px; }
.dia-meta a { color: var(--dia-accent); }
.dia-meta code {
  background: var(--dia-surface-alt);
  color: var(--dia-ink);
  border: 1px solid var(--dia-border);
  border-radius: 4px;
  padding: 0.1em 0.35em;
}

.dia-empty {
  padding: 24px;
  border: 1px dashed var(--dia-border);
  border-radius: 12px;
  background: var(--dia-bg);
  color: var(--dia-muted);
  text-align: center;
}

.gradio-container .prose code,
.gradio-container .prose :not(pre) > code {
  background: var(--dia-surface-alt);
  color: var(--dia-ink);
  border: 1px solid var(--dia-border);
  border-radius: 4px;
  padding: 0.1em 0.35em;
  font-size: 0.9em;
}

/* Plotly — dark-mode text + plot area (no JS; avoids breaking Gradio mount) */
:root.dark .gradio-container .js-plotly-plot .bg {
  fill: var(--dia-surface) !important;
}
:root.dark .gradio-container .js-plotly-plot .main-svg {
  background: var(--dia-surface);
}
:root.dark .gradio-container .js-plotly-plot text,
:root.dark .gradio-container .plotly text {
  fill: var(--dia-ink) !important;
}
:root.dark .gradio-container .js-plotly-plot .xaxislayer-above path,
:root.dark .gradio-container .js-plotly-plot .yaxislayer-above path {
  stroke: var(--dia-muted) !important;
}
"""


def dia_launch_kwargs() -> dict[str, object]:
    """Keyword args for ``Blocks.launch()`` (Gradio 6: theme/css belong here)."""
    return {"theme": dia_theme(), "css": DIA_CSS}
