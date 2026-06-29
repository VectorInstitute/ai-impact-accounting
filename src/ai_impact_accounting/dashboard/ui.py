"""Gradio dashboard.

Query a base model to see its family footprint, coverage, the base-vs-derivative
split (the paper's headline: derivatives can exceed the base), and a per-node
table coloured by data-quality tier.
"""

from __future__ import annotations

from typing import Any

import gradio as gr
import matplotlib as mpl
import pandas as pd


mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from ..graph import rollup  # noqa: E402


GREEN, BLUE, GREY = "#2e7d32", "#1565c0", "#bdbdbd"


def _bar(res: dict) -> Any:
    """Render the family carbon chart.

    Two honest modes, chosen by whether the base model itself disclosed a report:

    * **Base disclosed** → the paper's base-vs-derivatives comparison.
    * **Base undisclosed** → a per-derivative breakdown. We do NOT draw a base bar
      at zero, because an undisclosed base is *unknown*, not zero — a 0-height bar
      would imply the foundation model was free to train.
    """
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=120)
    bc, dc = res["base_footprint"]["carbon"], res["deriv_footprint"]["carbon"]
    base_disclosed = max(bc.lo, bc.hi) > 0
    derivs_disclosed = max(dc.lo, dc.hi) > 0

    if base_disclosed and derivs_disclosed:
        labels = ["Base model", "Derivatives\n(aggregate)"]
        # Normalize so lo<=hi. A degenerate/inverted interval would make an error
        # bar negative and matplotlib raises "yerr must not contain negative values".
        lows = [min(bc.lo, bc.hi), min(dc.lo, dc.hi)]
        highs = [max(bc.lo, bc.hi), max(dc.lo, dc.hi)]
        mids = [(lo + hi) / 2 for lo, hi in zip(lows, highs)]
        errs = [
            [max(0.0, m - lo) for m, lo in zip(mids, lows)],
            [max(0.0, hi - m) for m, hi in zip(highs, mids)],
        ]
        ax.bar(labels, mids, yerr=errs, capsize=6, color=[BLUE, GREEN], edgecolor="black", linewidth=0.6)
        ax.set_title(f"{res['base']}  —  base vs derivatives", fontsize=9)
    else:
        # No base-vs-derivative comparison possible (base undisclosed, or a root
        # with no derivatives yet). Show each disclosed model's own footprint —
        # one bar per model, never a phantom zero bar.
        disclosed = [r for r in res["rows"] if r["carbon_hi"] > 0]
        disclosed = sorted(disclosed, key=lambda r: r["carbon_hi"], reverse=True)
        labels = [r["model"].split("/")[-1] for r in disclosed]
        vals = [r["carbon_hi"] for r in disclosed]
        ax.bar(labels, vals, color=GREEN, edgecolor="black", linewidth=0.6)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
        suffix = "model footprint" if len(disclosed) == 1 else "per-model footprints (base undisclosed)"
        ax.set_title(f"{res['base']}  —  {suffix}", fontsize=9)

    ax.set_ylabel("Training CO₂ (kgCO₂eq)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _summary_md(res: dict) -> str:
    """Render the markdown summary block for a rollup result."""
    cov = res["coverage"] * 100
    head = (
        f"### {res['base']}\n"
        f"**Models in family:** {res['n_models']}  "
        f"({res['n_with_report']} with DIA report, {res['n_without_report']} missing)\n\n"
    )
    # A subtotal over a single disclosed node is not a family aggregate. Suppress
    # the number and say so. Floor is n<2 (a sample of one), NOT a coverage % --
    # the framework is built to report lower bounds at 10-40% coverage, so gating
    # on coverage would kill the number in exactly the regime it exists for.
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
    carbon_unit = " kgCO₂eq"
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
        f"| Carbon | {t['carbon'].fmt(carbon_unit)} |\n"
        f"| Water  | {t['water'].fmt(' L')} |\n"
        f"| Energy | {t['energy'].fmt(' kWh')} |\n\n" + prov_line + f"**Derivative footprint vs base:** {ratio_txt}\n"
    )


def _table(res: dict, row_filter: str = "all") -> tuple[Any, int, int]:
    """Build the per-model table dataframe and row counts."""
    rows = res["rows"]
    total = len(rows)
    if row_filter == "reporting":  # drop no-report ("—") rows
        rows = [r for r in rows if r["quality"] != "no report"]
    elif row_filter == "nonzero":  # also drop reported-zero / below-floor
        rows = [r for r in rows if r["carbon_hi"] > 0]
    df = pd.DataFrame(
        [
            {
                "Model": r["model"],
                "Role": r["role"],
                "Method": r["method"],
                "Carbon": r["carbon"],
                "Water": r["water"],
                "Energy": r["energy"],
                "Quality": r["quality"],
            }
            for r in rows
        ]
    )
    return df, len(rows), total


def render(store: Any, base: str, impute: bool, row_filter: str = "All") -> tuple[str, Any, Any]:
    """Render the dashboard outputs for a base model query.

    Parameters
    ----------
    store : Store
        The accounting store to read nodes from.
    base : str
        Base model id to roll up.
    impute : bool
        Whether to impute missing nodes from compute priors.
    row_filter : str, optional
        One of ``"All"``, ``"Reporting only"``, ``"Carbon > 0 only"``.

    Returns
    -------
    tuple
        ``(summary_markdown, matplotlib_figure_or_None, dataframe)``.
    """
    base = (base or "").strip()
    if not base:
        return "Enter a base model id, e.g. `meta-llama/Llama-3-8B`.", None, pd.DataFrame()
    res = rollup(store.nodes, base, impute=impute)
    mode = {"All": "all", "Reporting only": "reporting", "Carbon > 0 only": "nonzero"}.get(row_filter, "all")
    df, shown, total = _table(res, mode)
    disclosed = (res["n_with_report"] / res["n_models"] * 100) if res["n_models"] else 0.0
    note = f"\n\n*Table: showing {shown} of {total} models — {disclosed:.1f}% disclosed.*"
    # Draw a chart whenever at least one model disclosed: 2+ gives a family
    # comparison, exactly 1 still shows that model's own measured footprint so
    # selecting any base in the dropdown never leaves the chart blank.
    plot = _bar(res) if res["n_with_report"] >= 1 else None
    return _summary_md(res) + note, plot, df


def _base_choices(store: Any) -> list[str]:
    """List the rollup-able base/root ids for the dropdown.

    Picks the *parent* (foundation) model of each node so selecting an entry
    always rolls up a real family. A from-scratch node (parent ``"scratch"``) is
    a root in its own right, so it is offered by its own id instead. Leaf
    fine-tune repos are intentionally excluded (they would yield a 1-model
    "family" with no chart); a user can still type any id via custom value.
    """
    choices: set[str] = set()
    for mid, node in store.nodes.items():
        real_parents = [
            p.get("model")
            for p in (getattr(node, "lineage", []) or [])
            if p.get("model") and p.get("model") != "scratch"
        ]
        if real_parents:
            choices.update(real_parents)
        else:  # from-scratch root: it is its own queryable base
            choices.add(mid)
    return sorted(choices)


def build_ui(store: Any, default_base: str = "meta-llama/Llama-3-8B") -> Any:
    """Build the Gradio dashboard.

    Parameters
    ----------
    store : Store
        The accounting store backing the dashboard.
    default_base : str, optional
        Base model id pre-filled in the query box.

    Returns
    -------
    gradio.Blocks
        The assembled dashboard app.
    """
    with gr.Blocks(title="DIA — Data & Impact Accounting") as ui:
        gr.Markdown(
            "# \U0001f30d DIA — Data & Impact Accounting\n"
            "Cumulative carbon & water footprint across a model family and its "
            "derivatives. Push a model whose card carries a `dia_report` block and "
            "it appears here automatically."
        )
        base_options = _base_choices(store)
        if default_base and default_base not in base_options:
            base_options = [default_base, *base_options]
        with gr.Row():
            base_in = gr.Dropdown(
                label="Base model",
                choices=base_options,
                value=default_base,
                allow_custom_value=True,
                info="Pick a base/foundation model to roll up its derivatives, or type any id.",
                scale=4,
            )
            impute_in = gr.Checkbox(label="Impute missing nodes (labelled)", value=False)
            row_filter_in = gr.Radio(
                choices=["All", "Reporting only", "Carbon > 0 only"],
                value="All",
                label="Table rows",
                scale=2,
            )
            go = gr.Button("Roll up", variant="primary", scale=1)
        summary = gr.Markdown()
        with gr.Row():
            plot = gr.Plot(label="Base vs derivative")
        table = gr.Dataframe(label="Per-model footprints", wrap=True)
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
                "*unknown*, not zero, so no phantom zero bar is drawn."
            )

        def fn(b: str, i: bool, r: str) -> tuple[str, Any, Any]:
            return render(store, b, i, r)

        inputs = [base_in, impute_in, row_filter_in]
        go.click(fn, inputs, [summary, plot, table])
        base_in.change(fn, inputs, [summary, plot, table])
        row_filter_in.change(fn, inputs, [summary, plot, table])
        impute_in.change(fn, inputs, [summary, plot, table])
        ui.load(fn, inputs, [summary, plot, table])
    return ui
