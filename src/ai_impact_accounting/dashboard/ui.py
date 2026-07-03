"""Gradio dashboard.

Query a base model to see its family footprint, coverage, the base-vs-derivative
split (the paper's headline: derivatives can exceed the base), and a per-node
table coloured by data-quality tier.
"""

from __future__ import annotations

import html
import tempfile
from dataclasses import dataclass
from typing import Any, Optional

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from ..graph import rollup
from .theme import dia_launch_kwargs


BLUE = "#1e6a8a"
GREEN = "#2e7d52"
GREY = "#bdbdbd"
CHART_INK = "#1a1a1a"
CHART_GRID = "rgba(128,128,128,0.15)"
CHART_ZERO = "rgba(128,128,128,0.2)"


@dataclass
class RenderOutputs:
    """All dashboard widgets produced by a single render pass."""

    kpi_html: str
    summary_md: str
    plot: Optional[go.Figure]
    table_html: str
    hardware_md: str
    compare_md: str
    csv_path: Optional[str]


def _hf_url(model_id: str) -> str:
    return f"https://huggingface.co/{model_id}"


def _quality_class(quality: str) -> str:
    if quality == "no report":
        return "q-none"
    if quality == "measured":
        return "q-measured"
    if quality == "imputed":
        return "q-imputed"
    if quality.startswith("estimated"):
        return "q-estimated"
    return "q-none"


def _plotly_bar(res: dict) -> Optional[go.Figure]:
    """Interactive carbon chart (base vs derivatives or per-model breakdown)."""
    bc, dc = res["base_footprint"]["carbon"], res["deriv_footprint"]["carbon"]
    base_disclosed = max(bc.lo, bc.hi) > 0
    derivs_disclosed = max(dc.lo, dc.hi) > 0

    fig = go.Figure()
    if base_disclosed and derivs_disclosed:
        labels = ["Base model", "Derivatives"]
        lows = [min(bc.lo, bc.hi), min(dc.lo, dc.hi)]
        highs = [max(bc.lo, bc.hi), max(dc.lo, dc.hi)]
        mids = [(lo + hi) / 2 for lo, hi in zip(lows, highs, strict=True)]
        errs_plus = [max(0.0, hi - m) for m, hi in zip(mids, highs, strict=True)]
        errs_minus = [max(0.0, m - lo) for m, lo in zip(mids, lows, strict=True)]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=mids,
                error_y={"type": "data", "array": errs_plus, "arrayminus": errs_minus, "visible": True},
                marker_color=[BLUE, GREEN],
                marker_line={"color": "black", "width": 0.6},
                hovertemplate="%{x}<br>%{y:.4g} kgCO₂eq<extra></extra>",
            )
        )
        title = f"{res['base']} — base vs derivatives"
    else:
        disclosed = [r for r in res["rows"] if r["carbon_hi"] > 0]
        disclosed = sorted(disclosed, key=lambda r: r["carbon_hi"], reverse=True)
        if not disclosed:
            return None
        labels = [r["model"].split("/")[-1] for r in disclosed]
        vals = [r["carbon_hi"] for r in disclosed]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=vals,
                marker_color=GREEN,
                marker_line={"color": "black", "width": 0.6},
                hovertemplate="%{x}<br>%{y:.4g} kgCO₂eq<extra></extra>",
            )
        )
        suffix = "model footprint" if len(disclosed) == 1 else "per-model (base undisclosed)"
        title = f"{res['base']} — {suffix}"

    axis_font = {"color": CHART_INK}
    fig.update_layout(
        title={"text": title, "font": {"size": 13, "color": CHART_INK}},
        yaxis_title="Training CO₂ (kgCO₂eq)",
        margin={"l": 48, "r": 16, "t": 48, "b": 64},
        height=340,
        showlegend=False,
        paper_bgcolor="#faf9f7",
        plot_bgcolor="#ffffff",
        font=axis_font,
    )
    fig.update_xaxes(
        tickangle=-25,
        gridcolor=CHART_GRID,
        zerolinecolor=CHART_ZERO,
        tickfont=axis_font,
        title_font=axis_font,
        color=CHART_INK,
    )
    fig.update_yaxes(
        gridcolor=CHART_GRID,
        zerolinecolor=CHART_ZERO,
        tickfont=axis_font,
        title_font=axis_font,
        color=CHART_INK,
    )
    return fig


def _kpi_html(res: dict) -> str:
    """Top KPI cards for quick scanning."""
    cov = res["coverage"] * 100
    cov_warn = " ⚠️ lower bound" if cov < 60 else ""
    if res["n_with_report"] < 2:
        return f"""
        <div class="dia-kpi-row">
          <div class="dia-kpi"><div class="dia-kpi-label">Coverage</div>
            <div class="dia-kpi-value">{cov:.0f}%</div>
            <div class="dia-kpi-sub">{res['n_with_report']} of {res['n_models']} disclosed</div></div>
          <div class="dia-kpi"><div class="dia-kpi-label">Family size</div>
            <div class="dia-kpi-value">{res['n_models']}</div>
            <div class="dia-kpi-sub">models in subtree</div></div>
        </div>
        <div class="dia-empty">Insufficient disclosure to show family totals —
        per-model footprints only.</div>
        """

    t = res["total_footprint"]
    ratio = res["deriv_over_base_ratio"]
    ratio_txt = f"{ratio[0]:.1f}–{ratio[1]:.1f}×" if ratio else "n/a"

    return f"""
    <div class="dia-kpi-row">
      <div class="dia-kpi"><div class="dia-kpi-label">Coverage</div>
        <div class="dia-kpi-value">{cov:.0f}%</div>
        <div class="dia-kpi-sub">{res['n_with_report']}/{res['n_models']} models{cov_warn}</div></div>
      <div class="dia-kpi"><div class="dia-kpi-label">Carbon</div>
        <div class="dia-kpi-value">{html.escape(t['carbon'].fmt(''))}</div>
        <div class="dia-kpi-sub">kgCO₂eq disclosed subtotal</div></div>
      <div class="dia-kpi"><div class="dia-kpi-label">Water</div>
        <div class="dia-kpi-value">{html.escape(t['water'].fmt(''))}</div>
        <div class="dia-kpi-sub">litres</div></div>
      <div class="dia-kpi"><div class="dia-kpi-label">Energy</div>
        <div class="dia-kpi-value">{html.escape(t['energy'].fmt(''))}</div>
        <div class="dia-kpi-sub">kWh</div></div>
      <div class="dia-kpi"><div class="dia-kpi-label">Derivatives vs base</div>
        <div class="dia-kpi-value">{html.escape(ratio_txt)}</div>
        <div class="dia-kpi-sub">carbon ratio</div></div>
    </div>
    """


def _summary_md(res: dict) -> str:
    """Detailed markdown summary below KPIs."""
    cov = res["coverage"] * 100
    head = (
        f"### {res['base']}\n"
        f"**Models in family:** {res['n_models']}  "
        f"({res['n_with_report']} with DIA report, {res['n_without_report']} missing)\n\n"
    )
    if res["n_with_report"] < 2:
        return (
            head + f"**Coverage:** {cov:.0f}%\n\n"
            f"> ⚠️ Insufficient disclosure to aggregate "
            f"({res['n_with_report']} of {res['n_models']} models). "
            f"Per-model footprints only — no family total.\n"
        )
    t = res["total_footprint"]
    ratio = res["deriv_over_base_ratio"]
    ratio_txt = f"{ratio[0]:.1f}–{ratio[1]:.1f}× base" if ratio else "n/a (base footprint undisclosed)"
    flag = "" if cov >= 60 else "  ⚠️ low coverage — a LOWER BOUND"
    cbq = res.get("carbon_by_quality", {})
    prov = " · ".join(
        f"{tier} {iv.fmt(' kg')}"
        for tier, iv in (
            ("measured", cbq.get("measured")),
            ("estimated", cbq.get("estimated")),
            ("imputed", cbq.get("imputed")),
        )
        if iv and iv.hi > 0
    )
    prov_line = f"**Carbon provenance:** {prov}\n\n" if prov else ""
    return (
        head + f"**Coverage:** {cov:.0f}%{flag}\n\n"
        f"| Metric | Disclosed subtotal ({res['n_with_report']} of {res['n_models']}) |\n|---|---|\n"
        f"| Carbon | {t['carbon'].fmt(' kgCO₂eq')} |\n"
        f"| Water  | {t['water'].fmt(' L')} |\n"
        f"| Energy | {t['energy'].fmt(' kWh')} |\n\n" + prov_line + f"**Derivative footprint vs base:** {ratio_txt}\n"
    )


def _table_html(res: dict, row_filter: str = "all") -> tuple[str, pd.DataFrame, int, int]:
    """Styled HTML table with HF links and quality row colours."""
    rows = res["rows"]
    total = len(rows)
    if row_filter == "reporting":
        rows = [r for r in rows if r["quality"] != "no report"]
    elif row_filter == "nonzero":
        rows = [r for r in rows if r["carbon_hi"] > 0]

    body: list[str] = []
    records: list[dict[str, str]] = []
    for r in rows:
        mid = r["model"]
        link = f'<a href="{html.escape(_hf_url(mid))}" target="_blank" rel="noopener">{html.escape(mid)}</a>'
        qclass = _quality_class(r["quality"])
        body.append(
            f"<tr class='{qclass}'>"
            f"<td>{link}</td>"
            f"<td>{html.escape(r['role'])}</td>"
            f"<td>{html.escape(r['method'] or '—')}</td>"
            f"<td>{html.escape(r['carbon'])}</td>"
            f"<td>{html.escape(r['water'])}</td>"
            f"<td>{html.escape(r['energy'])}</td>"
            f"<td>{html.escape(r['quality'])}</td>"
            f"<td>{html.escape(r.get('gpu', '—'))}</td>"
            f"<td>{html.escape(r.get('region', '—'))}</td>"
            "</tr>"
        )
        records.append(
            {
                "Model": mid,
                "Role": r["role"],
                "Method": r["method"],
                "Carbon": r["carbon"],
                "Water": r["water"],
                "Energy": r["energy"],
                "Quality": r["quality"],
                "GPU": r.get("gpu", ""),
                "Region": r.get("region", ""),
            }
        )

    if not body:
        table = '<div class="dia-empty">No models match this filter.</div>'
    else:
        table = (
            "<table class='dia-table'><thead><tr>"
            "<th>Model</th><th>Role</th><th>Method</th>"
            "<th>Carbon</th><th>Water</th><th>Energy</th>"
            "<th>Quality</th><th>GPU</th><th>Region</th>"
            "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
        )
    df = pd.DataFrame(records)
    return table, df, len(rows), total


def _enrich_rows(res: dict, store: Any) -> dict:
    """Attach gpu/region from store nodes onto rollup rows."""
    enriched = dict(res)
    new_rows = []
    for r in res["rows"]:
        row = dict(r)
        node = store.nodes.get(r["model"])
        if node and node.report:
            row["gpu"] = node.report.gpu or "—"
            row["region"] = node.report.region or "—"
        else:
            row["gpu"] = "—"
            row["region"] = "—"
        new_rows.append(row)
    enriched["rows"] = new_rows
    return enriched


def _hardware_md(res: dict) -> str:
    """Hardware / region breakdown for disclosed models."""
    lines = ["### Hardware & region", ""]
    disclosed = [r for r in res["rows"] if r["quality"] != "no report"]
    if not disclosed:
        return "_No disclosed compute metadata for this family._"
    for r in disclosed:
        gpu = r.get("gpu", "—")
        region = r.get("region", "—")
        short = r["model"].split("/")[-1]
        lines.append(f"- **{short}** — `{gpu}`, region `{region}`")
    return "\n".join(lines)


def _compare_md(primary: dict, secondary: Optional[dict]) -> str:
    """Side-by-side comparison when a second base is selected."""
    if not secondary:
        return ""
    p_cov = primary["coverage"] * 100
    s_cov = secondary["coverage"] * 100
    p_c = primary["total_footprint"]["carbon"].fmt(" kg") if primary["n_with_report"] >= 2 else "n/a"
    s_c = secondary["total_footprint"]["carbon"].fmt(" kg") if secondary["n_with_report"] >= 2 else "n/a"
    return (
        f"### Compare families\n\n"
        f"| | **{primary['base']}** | **{secondary['base']}** |\n"
        f"|---|---|---|\n"
        f"| Models | {primary['n_models']} | {secondary['n_models']} |\n"
        f"| Disclosed | {primary['n_with_report']} | {secondary['n_with_report']} |\n"
        f"| Coverage | {p_cov:.0f}% | {s_cov:.0f}% |\n"
        f"| Carbon subtotal | {p_c} | {s_c} |\n"
    )


def _empty_outputs(message: str) -> RenderOutputs:
    return RenderOutputs(
        kpi_html=f'<div class="dia-empty">{html.escape(message)}</div>',
        summary_md=message,
        plot=None,
        table_html='<div class="dia-empty">No data to display.</div>',
        hardware_md="",
        compare_md="",
        csv_path=None,
    )


def render(
    store: Any,
    base: str,
    impute: bool,
    row_filter: str = "All",
    compare_base: str = "",
) -> RenderOutputs:
    """Render the dashboard for a base model query."""
    base = (base or "").strip()
    if not base:
        return _empty_outputs("Enter a base model id, e.g. `meta-llama/Llama-3-8B`.")

    if not store.nodes:
        return _empty_outputs(
            "Dataset is empty. Train a model with `dia_report`, ingest it, then refresh."
        )

    res = rollup(store.nodes, base, impute=impute)
    res = _enrich_rows(res, store)

    if res["n_models"] == 1 and res["n_with_report"] == 0:
        return _empty_outputs(
            f"No models in the dataset belong to family `{base}` yet, "
            "or none have disclosed a `dia_report`."
        )

    mode = {"All": "all", "Reporting only": "reporting", "Carbon > 0 only": "nonzero"}.get(row_filter, "all")
    table_html, df, shown, total = _table_html(res, mode)
    disclosed = (res["n_with_report"] / res["n_models"] * 100) if res["n_models"] else 0.0
    note = f"\n\n*Table: showing {shown} of {total} models — {disclosed:.1f}% disclosed.*"

    compare = ""
    if (compare_base or "").strip() and compare_base.strip() != base:
        sec = rollup(store.nodes, compare_base.strip(), impute=impute)
        sec = _enrich_rows(sec, store)
        compare = _compare_md(res, sec)

    csv_path: Optional[str] = None
    if not df.empty:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="dia-footprint-", mode="w") as tmp:
            df.to_csv(tmp.name, index=False)
            csv_path = tmp.name

    plot = _plotly_bar(res) if res["n_with_report"] >= 1 else None

    return RenderOutputs(
        kpi_html=_kpi_html(res),
        summary_md=_summary_md(res) + note,
        plot=plot,
        table_html=table_html,
        hardware_md=_hardware_md(res),
        compare_md=compare,
        csv_path=csv_path,
    )


def _dataset_header(store: Any) -> str:
    n = len(store.nodes)
    with_report = sum(1 for node in store.nodes.values() if node.has_report)
    return (
        f'<p class="dia-meta"><strong>Dataset:</strong> '
        f'<a href="https://huggingface.co/datasets/{html.escape(store.repo)}" '
        f'target="_blank" rel="noopener">{html.escape(store.repo)}</a>'
        f" &nbsp;·&nbsp; <strong>{n}</strong> node(s)"
        f" &nbsp;·&nbsp; <strong>{with_report}</strong> with <code>dia_report</code></p>"
    )


def _base_choices(store: Any) -> list[str]:
    """List rollup-able base/root ids for the dropdown."""
    choices: set[str] = set()
    for mid, node in store.nodes.items():
        real_parents = [
            p.get("model")
            for p in (getattr(node, "lineage", []) or [])
            if p.get("model") and p.get("model") != "scratch"
        ]
        if real_parents:
            choices.update(real_parents)
        else:
            choices.add(mid)
    return sorted(choices)


def _render_tuple(out: RenderOutputs) -> tuple:
    return (
        out.kpi_html,
        out.plot,
        out.summary_md,
        out.compare_md,
        out.table_html,
        out.hardware_md,
        out.csv_path,
    )


def build_ui(store: Any, default_base: str = "meta-llama/Llama-3-8B") -> Any:
    """Build the Gradio dashboard."""
    base_options = _base_choices(store)
    if default_base and default_base not in base_options:
        base_options = [default_base, *base_options]
    compare_choices = ["— none —", *base_options]

    with gr.Blocks(title="DIA — Data & Impact Accounting") as ui:
        gr.Markdown(
            "# 🌍 DIA — Data & Impact Accounting\n"
            "Cumulative carbon & water footprint across a model family and its "
            "derivatives. Models with a `dia_report` on their card appear after ingest."
        )
        meta_html = gr.HTML(_dataset_header(store))

        with gr.Row():
            base_in = gr.Dropdown(
                label="Base model",
                choices=base_options,
                value=default_base if default_base in base_options else (base_options[0] if base_options else None),
                allow_custom_value=True,
                info="Foundation checkpoint or family root to roll up.",
                scale=3,
            )
            compare_in = gr.Dropdown(
                label="Compare with (optional)",
                choices=compare_choices,
                value="— none —",
                allow_custom_value=True,
                info="Show a second family side-by-side.",
                scale=2,
            )
            refresh_btn = gr.Button("↻ Refresh data", scale=1)

        with gr.Row():
            impute_in = gr.Checkbox(
                label="Impute missing nodes",
                value=False,
                info="Fill undisclosed models with labelled priors (experimental).",
            )
            row_filter_in = gr.Radio(
                choices=["All", "Reporting only", "Carbon > 0 only"],
                value="All",
                label="Table rows",
                info="Reporting only hides models without a DIA report.",
            )
            go = gr.Button("View family footprint", variant="primary")

        kpi = gr.HTML()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                plot = gr.Plot(label="Carbon footprint", show_label=True)
            with gr.Column(scale=2):
                summary = gr.Markdown()
        compare_md = gr.Markdown()
        table_html = gr.HTML(label="Per-model footprints")
        hardware = gr.Markdown()
        csv_file = gr.File(label="Export CSV", interactive=False)

        with gr.Accordion("How the footprint is computed", open=False):
            gr.Markdown(
                "**Rollup method** — given a base model, we build the lineage as a "
                "directed graph (parent → child) and take the base plus all its "
                "descendants as the *family*.\n\n"
                "1. **Sum incremental footprints.** Each model reports only its own "
                "training delta; the family total is the sum over the subtree.\n"
                "2. **Dedupe the DAG.** A merged/shared model with multiple parents is "
                "counted **once** (visited-set traversal), so nothing is double-counted.\n"
                "3. **Report coverage, not a bare total.** We show *X% disclosed* — the "
                "total is a **lower bound** when reports are missing.\n"
                "4. **Keep provenance separate.** Carbon is split into `measured` "
                "(CodeCarbon/NVML) vs `estimated` vs `imputed`, never mixed.\n\n"
                "*Chart:* if the base model itself disclosed a report you get a "
                "**base-vs-derivatives** comparison; otherwise (foundation base not "
                "measured) you get a **per-model breakdown** — an undisclosed base is "
                "*unknown*, not zero, so no phantom zero bar is drawn.\n\n"
                "**Table colours:** green = measured · amber = estimated · grey = "
                "imputed / no report."
            )

        outputs = [kpi, plot, summary, compare_md, table_html, hardware, csv_file]

        def _compare_value(raw: str) -> str:
            return "" if raw in ("", "— none —", None) else raw.strip()

        def fn(b: str, cmp_base: str, i: bool, r: str) -> tuple:
            return _render_tuple(render(store, b, i, r, _compare_value(cmp_base)))

        inputs = [base_in, compare_in, impute_in, row_filter_in]

        def refresh(b: str, cmp_base: str, i: bool, r: str) -> tuple:
            store.load()
            choices = _base_choices(store)
            cmp_choices = ["— none —", *choices]
            out = render(store, b, i, r, _compare_value(cmp_base))
            return (
                gr.Dropdown(choices=choices),
                gr.Dropdown(choices=cmp_choices),
                _dataset_header(store),
                *_render_tuple(out),
            )

        go.click(fn, inputs, outputs)
        base_in.change(fn, inputs, outputs)
        compare_in.change(fn, inputs, outputs)
        row_filter_in.change(fn, inputs, outputs)
        impute_in.change(fn, inputs, outputs)
        ui.load(fn, inputs, outputs)

        refresh_btn.click(
            refresh,
            inputs,
            [base_in, compare_in, meta_html, *outputs],
        )

    return ui


__all__ = ["build_ui", "dia_launch_kwargs", "render"]
