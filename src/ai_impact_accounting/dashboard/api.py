"""JSON API for the DIA web dashboard (Gradio-free)."""

from __future__ import annotations

import csv
import io
from typing import Any, Literal, Optional

from ..graph import rollup
from ..hub import ingest_model
from ..hub.ingest import fetch_model_card
from ..parse import parse_card_disclosure, parse_lineage, parse_report
from .atlas import graph_payload


RowFilter = Literal["all", "reporting", "nonzero"]
GraphView = Literal["all", "family"]

QUALITY_COLORS = {
    "measured": "#2e7d52",
    "estimated": "#c9a017",
    "imputed": "#9ca3af",
    "no report": "#cbd5e1",
    "not in dataset": "#e8e4dc",
    "placeholder": "#94a3b8",
    "disclosed-on-card": "#0d9488",
}

RELATION_COLORS = {
    "finetune": "#3d9a40",
    "adapter": "#0ea5b7",
    "lora": "#0ea5b7",
    "qlora": "#0891b2",
    "quantized": "#9333ea",
    "merge": "#1e40af",
    "distill": "#65a30d",
    "fork": "#78716c",
}

GRAPH_LEGEND = {
    "nodes": [
        {"key": "measured", "label": "Measured footprint", "color": "#2e7d52"},
        {"key": "estimated", "label": "Estimated footprint", "color": "#c9a017"},
        {"key": "imputed", "label": "Imputed footprint", "color": "#9ca3af"},
        {"key": "no report", "label": "No DIA report", "color": "#cbd5e1"},
        {"key": "not in dataset", "label": "Lineage parent only", "color": "#e8e4dc"},
        {"key": "placeholder", "label": "From scratch (placeholder)", "color": "#94a3b8"},
        {"key": "disclosed-on-card", "label": "Publisher disclosure (model card)", "color": "#0d9488"},
    ],
    "edges": [
        {"key": "finetune", "label": "finetune", "color": "#3d9a40"},
        {"key": "adapter", "label": "adapter / LoRA", "color": "#0ea5b7"},
        {"key": "quantized", "label": "quantized", "color": "#9333ea"},
        {"key": "merge", "label": "merge", "color": "#1e40af"},
        {"key": "other", "label": "other relation", "color": "#94a3b8"},
    ],
}


def _quality_color_key(quality: str) -> str:
    if quality == "placeholder":
        return "placeholder"
    if quality == "disclosed-on-card":
        return "disclosed-on-card"
    if quality == "not in dataset":
        return "not in dataset"
    if quality == "no report":
        return "no report"
    if quality == "measured":
        return "measured"
    if quality == "imputed":
        return "imputed"
    if quality.startswith("estimated"):
        return "estimated"
    return "no report"


def _relation_color(relation: str) -> str:
    rel = (relation or "finetune").lower()
    return RELATION_COLORS.get(rel, RELATION_COLORS.get("fork", "#94a3b8"))


def is_scratch_sentinel(model_id: str) -> bool:
    """True for the from-scratch lineage placeholder, not a real Hub repo."""
    return (model_id or "").strip() == "scratch"


def is_hub_model_id(model_id: str) -> bool:
    """True when id looks like a Hugging Face repo (org/name), not a sentinel like scratch."""
    mid = (model_id or "").strip()
    if not mid or is_scratch_sentinel(mid):
        return False
    return "/" in mid and "://" not in mid


def hf_url(model_id: str) -> str | None:
    """Return the Hugging Face model page URL, or None for non-Hub ids."""
    if not is_hub_model_id(model_id):
        return None
    return f"https://huggingface.co/{model_id}"


_card_disclosure_cache: dict[str, dict[str, Any] | None] = {}


def clear_card_disclosure_cache() -> None:
    """Drop cached Hub card carbon parses (e.g. after dataset refresh)."""
    _card_disclosure_cache.clear()


def get_card_disclosure(model_id: str, token: str | None = None) -> dict[str, Any] | None:
    """Fetch and parse publisher carbon disclosure for one Hub model."""
    if not is_hub_model_id(model_id):
        return None
    if model_id in _card_disclosure_cache:
        return _card_disclosure_cache[model_id]
    try:
        _, card_text = fetch_model_card(model_id, token=token)
        disc = parse_card_disclosure(card_text, model_id)
    except Exception:
        disc = None
    _card_disclosure_cache[model_id] = disc
    return disc


def get_card_disclosures(model_ids: list[str], token: str | None = None) -> dict[str, dict[str, Any]]:
    """Batch-fetch publisher disclosures; skips ids already cached."""
    out: dict[str, dict[str, Any]] = {}
    for mid in model_ids:
        disc = get_card_disclosure(mid, token=token)
        if disc:
            out[mid] = disc
    return out


def _card_lookup_candidates(
    store_nodes: dict,
    graph_nodes: list[dict[str, Any]],
) -> list[str]:
    """Hub model ids that may have publisher carbon on their card."""
    ids: set[str] = set()
    for n in graph_nodes:
        mid = n["id"]
        if not is_hub_model_id(mid):
            continue
        node = store_nodes.get(mid)
        if n.get("quality") in ("not in dataset", "no report") and (node is None or not node.has_report):
            ids.add(mid)
    return sorted(ids)


def base_choices(store: Any) -> list[str]:
    """List rollup-able base/root ids."""
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


def dataset_meta(store: Any) -> dict[str, Any]:
    """Summarize the loaded dataset for the UI header."""
    n = len(store.nodes)
    with_report = sum(1 for node in store.nodes.values() if node.has_report)
    return {
        "dataset": store.repo,
        "dataset_url": f"https://huggingface.co/datasets/{store.repo}",
        "n_nodes": n,
        "n_with_report": with_report,
    }


def _enrich_rows(res: dict, store: Any) -> dict:
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


def _quality_key(quality: str) -> str:
    if quality == "no report":
        return "none"
    if quality == "measured":
        return "measured"
    if quality == "imputed":
        return "imputed"
    if quality.startswith("estimated"):
        return "estimated"
    return "none"


def _interval_json(iv: Any) -> dict[str, float]:
    return {"lo": iv.lo, "hi": iv.hi, "fmt": iv.fmt("")}


def _rollup_json(res: dict, *, base_card: dict[str, Any] | None = None) -> dict[str, Any]:
    t = res["total_footprint"]
    ratio = res["deriv_over_base_ratio"]
    cbq = res.get("carbon_by_quality", {})
    return {
        "base": res["base"],
        "n_models": res["n_models"],
        "n_with_report": res["n_with_report"],
        "n_without_report": res["n_without_report"],
        "coverage": res["coverage"],
        "base_card_disclosure": base_card,
        "total_footprint": {
            "carbon": _interval_json(t["carbon"]),
            "water": _interval_json(t["water"]),
            "energy": _interval_json(t["energy"]),
        },
        "base_footprint": {"carbon": _interval_json(res["base_footprint"]["carbon"])},
        "deriv_footprint": {"carbon": _interval_json(res["deriv_footprint"]["carbon"])},
        "deriv_over_base_ratio": list(ratio) if ratio else None,
        "carbon_by_quality": {
            tier: _interval_json(iv)
            for tier, iv in cbq.items()
            if iv and iv.hi > 0
        },
        "rows": [
            {
                "model": r["model"],
                "model_url": hf_url(r["model"]),
                "role": r["role"],
                "method": r["method"] or "",
                "carbon": r["carbon"],
                "carbon_hi": r["carbon_hi"],
                "water": r["water"],
                "energy": r["energy"],
                "quality": r["quality"],
                "quality_key": _quality_key(r["quality"]),
                "gpu": r.get("gpu", "—"),
                "region": r.get("region", "—"),
            }
            for r in res["rows"]
        ],
    }


def _filter_rows(rows: list[dict], row_filter: RowFilter) -> list[dict]:
    if row_filter == "reporting":
        return [r for r in rows if r["quality"] != "no report"]
    if row_filter == "nonzero":
        return [r for r in rows if r["carbon_hi"] > 0]
    return rows


def bar_chart_data(res: dict) -> Optional[dict[str, Any]]:
    """Carbon bar chart spec for Chart.js."""
    bc, dc = res["base_footprint"]["carbon"], res["deriv_footprint"]["carbon"]
    base_disclosed = max(bc.lo, bc.hi) > 0
    derivs_disclosed = max(dc.lo, dc.hi) > 0

    if base_disclosed and derivs_disclosed:
        return {
            "kind": "base_vs_deriv",
            "title": f"{res['base']} — base vs derivatives",
            "labels": ["Base model", "Derivatives"],
            "values": [
                (bc.lo + bc.hi) / 2,
                (dc.lo + dc.hi) / 2,
            ],
            "errors_plus": [max(0.0, bc.hi - (bc.lo + bc.hi) / 2), max(0.0, dc.hi - (dc.lo + dc.hi) / 2)],
            "errors_minus": [max(0.0, (bc.lo + bc.hi) / 2 - bc.lo), max(0.0, (dc.lo + dc.hi) / 2 - dc.lo)],
            "colors": ["#1e6a8a", "#2e7d52"],
        }

    disclosed = [r for r in res["rows"] if r["carbon_hi"] > 0]
    disclosed = sorted(disclosed, key=lambda r: r["carbon_hi"], reverse=True)
    if not disclosed:
        return None
    suffix = "model footprint" if len(disclosed) == 1 else "per-model (base undisclosed)"
    return {
        "kind": "per_model",
        "title": f"{res['base']} — {suffix}",
        "labels": [r["model"].split("/")[-1] for r in disclosed],
        "values": [r["carbon_hi"] for r in disclosed],
        "colors": ["#2e7d52"] * len(disclosed),
    }


def compare_families(primary: dict, secondary: dict) -> dict[str, Any]:
    """Side-by-side comparison table for two family rollups."""
    p_cov = primary["coverage"] * 100
    s_cov = secondary["coverage"] * 100
    p_c = (
        primary["total_footprint"]["carbon"]["fmt"]
        if primary["n_with_report"] >= 2
        else "n/a"
    )
    s_c = (
        secondary["total_footprint"]["carbon"]["fmt"]
        if secondary["n_with_report"] >= 2
        else "n/a"
    )
    return {
        "primary_base": primary["base"],
        "secondary_base": secondary["base"],
        "rows": [
            {"label": "Models", "primary": primary["n_models"], "secondary": secondary["n_models"]},
            {"label": "Disclosed", "primary": primary["n_with_report"], "secondary": secondary["n_with_report"]},
            {"label": "Coverage", "primary": f"{p_cov:.0f}%", "secondary": f"{s_cov:.0f}%"},
            {"label": "Carbon subtotal", "primary": p_c, "secondary": s_c},
        ],
    }


def graph_vis_payload(
    nodes: dict,
    base: str,
    view: GraphView = "all",
    impute: bool = False,
    rollup_res: Optional[dict] = None,
    card_disclosures: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Graph nodes/edges formatted for vis-network."""
    payload = graph_payload(nodes, base, view=view, impute=impute, rollup_res=rollup_res)
    disclosures = card_disclosures or {}
    vis_nodes = []
    for n in payload["nodes"]:
        quality = n.get("quality", "no report")
        carbon_display = n.get("carbon", "—")
        disc = disclosures.get(n["id"])
        if disc:
            quality = "disclosed-on-card"
            carbon_display = disc["carbon"]
        qkey = _quality_color_key(quality)
        color = QUALITY_COLORS.get(qkey, QUALITY_COLORS["no report"])
        is_placeholder = is_scratch_sentinel(n["id"])
        try:
            if disc and disc.get("carbon_kg"):
                size = 22 + min(float(disc["carbon_kg"]) / 20000.0, 12)
            else:
                size = 22 + min(float(str(carbon_display).split()[0].replace(",", "")) * 8000, 10)
        except ValueError:
            size = 22
        if n.get("highlighted"):
            size += 6
        if is_placeholder:
            size = 26
        if is_placeholder:
            title = (
                "From scratch (placeholder)\n"
                "Not a Hugging Face model — groups models trained without a parent checkpoint."
            )
        elif disc:
            title = (
                f"{n['id']}\nRole: {n.get('role', '')}\n"
                f"Carbon: {carbon_display} (model card, {disc.get('scope', 'pretraining')})\n"
                f"GPU hours: {disc.get('gpu_hours', '—')}\n"
                f"Hardware: {disc.get('hardware', '—')}\n"
                f"Quality: publisher disclosure (not dia_report)"
            )
        else:
            title = (
                f"{n['label']}\nRole: {n.get('role', '')}\n"
                f"Carbon: {carbon_display}\n"
                f"Water: {n.get('water', '—')}\n"
                f"Quality: {quality}"
            )
        vis_node: dict[str, Any] = {
            "id": n["id"],
            "label": n["label"],
            "display_label": n["label"] if is_placeholder else n["id"],
            "hub_url": None if is_placeholder else hf_url(n["id"]),
            "is_placeholder": is_placeholder,
            "x": n.get("x"),
            "y": n.get("y"),
            "fixed": {"x": True, "y": True},
            "quality_key": qkey,
            "title": title,
            "role": n.get("role", ""),
            "water": n.get("water", "—"),
            "quality": quality,
            "carbon": carbon_display,
            "card_disclosure": disc,
            "color": {
                "background": color,
                "border": "#64748b" if is_placeholder else ("#1a5c35" if n.get("highlighted") else "#555"),
                "highlight": {"background": color, "border": "#1e6a8a"},
            },
            "size": size,
            "font": {
                "size": 16,
                "color": "#1a1a1a",
                "strokeWidth": 5,
                "strokeColor": "#ffffff",
                "face": "Segoe UI, system-ui, sans-serif",
            },
        }
        if is_placeholder:
            vis_node["shapeProperties"] = {"borderDashes": [6, 4]}
        vis_nodes.append(vis_node)
    vis_edges = []
    for e in payload["edges"]:
        rel = e.get("relation") or "finetune"
        edge_color = _relation_color(rel)
        vis_edges.append(
            {
                "from": e["source"],
                "to": e["target"],
                "title": rel,
                "arrows": "to",
                "relation": rel,
                "color": {"color": edge_color, "highlight": "#1e6a8a", "opacity": 0.85},
                "width": 2.5,
            }
        )
    return {
        "view": payload["view"],
        "base": payload["base"],
        "scope_label": payload["scope_label"],
        "n_models": payload["n_models"],
        "n_edges": payload["n_edges"],
        "coverage": payload["coverage"],
        "nodes": vis_nodes,
        "edges": vis_edges,
        "legend": GRAPH_LEGEND,
    }


def kpi_cards(res: dict, *, base_card: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Top KPI card values for the dashboard header row."""
    cov = res["coverage"] * 100
    cards: list[dict[str, str]] = []

    if res["n_with_report"] >= 2:
        t = res["total_footprint"]
        ratio = res["deriv_over_base_ratio"]
        ratio_txt = f"{ratio[0]:.1f}–{ratio[1]:.1f}×" if ratio else "n/a"
        cards.extend(
            [
                {
                    "label": "Family carbon",
                    "value": f"{t['carbon']['fmt']} kg",
                    "sub": "disclosed subtotal",
                    "primary": "true",
                },
                {"label": "Water", "value": t["water"]["fmt"], "sub": "litres"},
                {"label": "Energy", "value": t["energy"]["fmt"], "sub": "kWh"},
                {"label": "Derivatives vs base", "value": ratio_txt, "sub": "carbon ratio"},
            ]
        )

    cards.extend(
        [
            {
                "label": "Coverage",
                "value": f"{cov:.0f}%",
                "sub": f"{res['n_with_report']}/{res['n_models']} with footprint data",
            },
            {
                "label": "Family size",
                "value": str(res["n_models"]),
                "sub": "models in subtree",
            },
        ]
    )

    if base_card:
        cards.append(
            {
                "label": "Base pretraining (card)",
                "value": base_card["carbon"],
                "sub": f"{base_card.get('variant', 'publisher')} · not in rollup",
            }
        )
        if base_card.get("gpu_hours"):
            cards.append(
                {
                    "label": "Base GPU hours (card)",
                    "value": str(base_card["gpu_hours"]),
                    "sub": base_card.get("hardware") or "from model card",
                }
            )
    return cards


def dashboard_payload(
    store: Any,
    base: str,
    impute: bool = False,
    row_filter: RowFilter = "all",
    compare_base: str = "",
    graph_view: GraphView = "all",
) -> dict[str, Any]:
    """Full dashboard state for one query."""
    base = (base or "").strip()
    if not base:
        return {"ok": False, "error": "Enter a base model id, e.g. `meta-llama/Llama-3-8B`."}
    if not store.nodes:
        return {
            "ok": False,
            "error": "Dataset is empty. Train a model with `dia_report`, ingest it, then refresh.",
        }

    res = rollup(store.nodes, base, impute=impute)
    res = _enrich_rows(res, store)

    graph_raw = graph_payload(
        store.nodes,
        base,
        view=graph_view,
        impute=impute,
        rollup_res=res if graph_view == "family" else None,
    )
    token = getattr(store, "token", None)
    card_disclosures = get_card_disclosures(_card_lookup_candidates(store.nodes, graph_raw["nodes"]), token)
    base_card = card_disclosures.get(base)
    rollup_data = _rollup_json(res, base_card=base_card)

    if res["n_models"] == 1 and res["n_with_report"] == 0:
        return {
            "ok": False,
            "error": (
                f"No models in the dataset belong to family `{base}` yet, "
                "or none have disclosed a `dia_report`."
            ),
        }

    filtered = _filter_rows(rollup_data["rows"], row_filter)
    compare = None
    cmp = (compare_base or "").strip()
    if cmp and cmp != base:
        sec = rollup(store.nodes, cmp, impute=impute)
        sec = _enrich_rows(sec, store)
        compare = compare_families(_rollup_json(res), _rollup_json(sec))

    hardware = [
        {
            "model": r["model"],
            "short": r["model"].split("/")[-1],
            "gpu": r["gpu"],
            "region": r["region"],
        }
        for r in rollup_data["rows"]
        if r["quality"] != "no report"
    ]

    graph = graph_vis_payload(
        store.nodes,
        base,
        view=graph_view,
        impute=impute,
        rollup_res=res if graph_view == "family" else None,
        card_disclosures=card_disclosures,
    )

    return {
        "ok": True,
        "meta": dataset_meta(store),
        "base": base,
        "impute": impute,
        "row_filter": row_filter,
        "graph_view": graph_view,
        "rollup": rollup_data,
        "table_rows": filtered,
        "table_total": len(rollup_data["rows"]),
        "table_shown": len(filtered),
        "kpi": kpi_cards(rollup_data, base_card=base_card),
        "bar_chart": bar_chart_data(res),
        "hardware": hardware,
        "compare": compare,
        "graph": graph,
        "drill_choices": sorted(n["id"] for n in graph_payload(store.nodes, base, view="all")["nodes"]),
    }


def _report_snapshot(rep: Any) -> dict[str, Any]:
    return {
        "carbon": rep.carbon.fmt(" kgCO₂eq"),
        "water": rep.water.fmt(" L"),
        "energy": rep.energy.fmt(" kWh"),
        "quality": rep.quality.get("carbon", ""),
        "method": rep.method or "",
        "gpu": rep.gpu or "",
        "region": rep.region or "",
    }


def hub_lookup(store: Any, model_id: str) -> dict[str, Any]:
    """Fetch a model card from the Hub and compare with local store state."""
    model_id = (model_id or "").strip()
    if not model_id:
        return {"ok": False, "error": "Model id is required."}
    if is_scratch_sentinel(model_id):
        return {
            "ok": False,
            "model": model_id,
            "model_url": None,
            "error": "Not a Hugging Face model",
            "message": "Trained from scratch — there is no parent model on the Hub.",
        }

    stored = store.nodes.get(model_id)
    hub_link = hf_url(model_id)
    result: dict[str, Any] = {
        "ok": True,
        "model": model_id,
        "model_url": hub_link,
        "in_store": stored is not None,
        "in_store_has_report": bool(stored and stored.has_report),
        "store_footprint": _report_snapshot(stored.report) if stored and stored.report else None,
        "writable": bool(getattr(store, "token", None)),
    }

    try:
        meta, card_text = fetch_model_card(model_id, token=getattr(store, "token", None))
    except Exception as exc:
        return {
            "ok": False,
            "model": model_id,
            "model_url": hub_link,
            "error": type(exc).__name__,
            "message": "Could not load model card from Hugging Face.",
        }

    report, errors = parse_report(meta)
    lineage = parse_lineage(meta)
    card_disclosure = parse_card_disclosure(card_text, model_id) if report is None else None
    if card_disclosure:
        _card_disclosure_cache[model_id] = card_disclosure
    result.update(
        {
            "hub_has_report": report is not None,
            "hub_footprint": _report_snapshot(report) if report else None,
            "card_disclosure": card_disclosure,
            "lineage": lineage,
            "parse_errors": errors,
            "can_ingest": bool(getattr(store, "token", None)) and report is not None,
        }
    )
    return result


def hub_ingest(store: Any, model_id: str) -> dict[str, Any]:
    """Ingest one model from the Hub into the local store."""
    model_id = (model_id or "").strip()
    if not model_id:
        return {"ok": False, "error": "Model id is required."}
    if is_scratch_sentinel(model_id) or not is_hub_model_id(model_id):
        return {
            "ok": False,
            "error": "Not a Hugging Face model id — nothing to ingest from the Hub.",
        }
    if not getattr(store, "token", None):
        return {
            "ok": False,
            "error": "Write token required (HF_TOKEN) to ingest into the dataset.",
        }

    res = ingest_model(model_id, store, token=store.token, persist=True)
    if not res.get("ok"):
        return res
    return {
        "ok": True,
        "model": model_id,
        "has_report": res.get("has_report", False),
        "lineage": res.get("lineage", []),
        "errors": res.get("errors", []),
    }


def export_csv(
    store: Any,
    base: str,
    impute: bool = False,
    row_filter: RowFilter = "all",
) -> str:
    """Return CSV text for the current family table."""
    payload = dashboard_payload(store, base, impute=impute, row_filter=row_filter, graph_view="family")
    if not payload.get("ok"):
        return "Model,Error\n," + payload.get("error", "unknown")
    buf = io.StringIO()
    fields = ["Model", "Role", "Method", "Carbon", "Water", "Energy", "Quality", "GPU", "Region"]
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for r in payload["table_rows"]:
        writer.writerow(
            {
                "Model": r["model"],
                "Role": r["role"],
                "Method": r["method"],
                "Carbon": r["carbon"],
                "Water": r["water"],
                "Energy": r["energy"],
                "Quality": r["quality"],
                "GPU": r["gpu"],
                "Region": r["region"],
            }
        )
    return buf.getvalue()
