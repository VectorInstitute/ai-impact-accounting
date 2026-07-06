"""Tests for the web dashboard JSON API."""

from unittest.mock import patch

from ai_impact_accounting import Node, Report
from ai_impact_accounting.dashboard.api import (
    bar_chart_data,
    base_choices,
    dashboard_payload,
    graph_vis_payload,
    kpi_cards,
    _rollup_json,
)
from ai_impact_accounting.graph import rollup
from ai_impact_accounting.models import Interval


class _FakeStore:
    def __init__(self, nodes: dict, repo: str = "test/dia-state") -> None:
        self.nodes = nodes
        self.repo = repo


def _node(mid, carbon=None, parents=()):
    rep = None
    if carbon is not None:
        rep = Report(scope="incremental")
        rep.carbon = Interval(carbon, carbon)
        rep.energy = Interval(carbon, carbon)
        rep.quality = {"carbon": "measured"}
    lineage = [{"model": p, "relation": "finetune"} for p in parents]
    return Node(model_id=mid, report=rep, lineage=lineage)


def test_dashboard_payload_ok():
    nodes = {
        "B": _node("B", carbon=1.0),
        "D1": _node("D1", carbon=0.1, parents=["B"]),
    }
    store = _FakeStore(nodes)
    out = dashboard_payload(store, "B")
    assert out["ok"] is True
    assert out["rollup"]["n_models"] == 2
    assert len(out["kpi"]) >= 2
    assert out["graph"]["n_models"] == 2


def test_base_choices_includes_lineage_parents():
    nodes = {"D1": _node("D1", carbon=0.1, parents=["meta-llama/Llama-3-8B"])}
    store = _FakeStore(nodes)
    assert "meta-llama/Llama-3-8B" in base_choices(store)


def test_graph_vis_payload_has_vis_nodes():
    nodes = {
        "B": _node("B", carbon=1.0),
        "D1": _node("D1", carbon=0.1, parents=["B"]),
    }
    payload = graph_vis_payload(nodes, "B", view="all")
    assert payload["nodes"][0]["color"]["background"]
    assert payload["edges"][0]["from"] == "B"
    assert "legend" in payload
    assert len(payload["legend"]["nodes"]) >= 4


def test_bar_chart_base_vs_deriv():
    nodes = {
        "B": _node("B", carbon=1.0),
        "D1": _node("D1", carbon=0.2, parents=["B"]),
    }
    res = rollup(nodes, "B")
    chart = bar_chart_data(res)
    assert chart is not None
    assert chart["kind"] == "base_vs_deriv"


def test_graph_enriches_lineage_parent_with_card_disclosure():
    parent_id = "meta-llama/Llama-3.2-3B-Instruct"
    disc = {
        parent_id: {
            "carbon": "133,000 kgCO₂eq (location-based)",
            "carbon_tonnes": 133.0,
            "carbon_kg": 133_000.0,
            "gpu_hours": "460k",
            "hardware": "H100-80GB",
            "scope": "pretraining",
            "variant": "Llama 3.2 3B",
        }
    }
    nodes = {"D1": _node("D1", carbon=0.1, parents=[parent_id])}
    payload = graph_vis_payload(nodes, parent_id, view="all", card_disclosures=disc)
    parent = next(n for n in payload["nodes"] if n["id"] == parent_id)
    assert parent["quality_key"] == "disclosed-on-card"
    assert "133,000" in parent["title"]
    assert parent["card_disclosure"]["gpu_hours"] == "460k"


@patch("ai_impact_accounting.dashboard.api.get_card_disclosures")
def test_dashboard_kpi_includes_base_card_disclosure(mock_disc):
    parent_id = "meta-llama/Llama-3.2-3B-Instruct"
    mock_disc.return_value = {
        parent_id: {
            "carbon": "133,000 kgCO₂eq",
            "carbon_kg": 133_000.0,
            "gpu_hours": "460k",
            "hardware": "H100-80GB",
            "variant": "Llama 3.2 3B",
        }
    }
    nodes = {"D1": _node("D1", carbon=0.1, parents=[parent_id])}
    store = _FakeStore(nodes)
    out = dashboard_payload(store, parent_id)
    assert out["ok"] is True
    assert out["rollup"]["base_card_disclosure"]["carbon"] == "133,000 kgCO₂eq"
    labels = [c["label"] for c in out["kpi"]]
    assert "Base carbon (card)" in labels


def test_kpi_cards_base_card_when_few_dia_reports():
    res = rollup({"D1": _node("D1", carbon=0.1, parents=["meta-llama/Llama-3.2-3B-Instruct"])},
                 "meta-llama/Llama-3.2-3B-Instruct")
    rollup_json = _rollup_json(res, base_card={"carbon": "133,000 kgCO₂eq", "gpu_hours": "460k", "variant": "Llama 3.2 3B"})
    cards = kpi_cards(rollup_json, base_card=rollup_json["base_card_disclosure"])
    assert any(c["label"] == "Base carbon (card)" for c in cards)
