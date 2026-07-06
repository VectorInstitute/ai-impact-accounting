"""Tests for the impact lineage graph payload and figure."""

from pathlib import Path

from ai_impact_accounting import LocalStore, Node, Report
from ai_impact_accounting.dashboard.atlas import graph_payload, impact_graph_figure
from ai_impact_accounting.models import Interval


def _node(mid, carbon=None, parents=()):
    rep = None
    if carbon is not None:
        rep = Report(scope="incremental")
        rep.carbon = Interval(carbon, carbon)
        rep.energy = Interval(carbon, carbon)
        rep.quality = {"carbon": "measured"}
    lineage = [{"model": p, "relation": "finetune"} for p in parents]
    return Node(model_id=mid, report=rep, lineage=lineage)


def test_family_graph_payload_structure():
    nodes = {
        "B": _node("B", carbon=1.0),
        "D1": _node("D1", carbon=0.1, parents=["B"]),
        "D2": _node("D2", carbon=0.2, parents=["B"]),
    }
    payload = graph_payload(nodes, "B", view="family")
    assert payload["view"] == "family"
    assert payload["n_models"] == 3
    assert payload["n_edges"] == 2
    assert {n["id"] for n in payload["nodes"]} == {"B", "D1", "D2"}


def test_dataset_graph_matches_full_dag():
    nodes = {
        "B": _node("B", carbon=1.0),
        "D1": _node("D1", carbon=0.1, parents=["B"]),
        "D2": _node("D2", carbon=0.2, parents=["B"]),
    }
    family = graph_payload(nodes, "B", view="family")
    full = graph_payload(nodes, "B", view="all")
    assert full["view"] == "all"
    assert full["n_models"] == family["n_models"]
    assert full["n_edges"] == family["n_edges"]


def test_impact_graph_figure_all_models_default():
    nodes = {
        "B": _node("B", carbon=1.0),
        "D1": _node("D1", carbon=0.1, parents=["B"]),
    }
    fig = impact_graph_figure(nodes, "B", view="all")
    assert fig is not None
    assert "Full dataset" in fig.layout.title.text


def test_layered_layout_spreads_left_to_right():
    nodes = {
        "B": _node("B", carbon=1.0),
        "D1": _node("D1", carbon=0.1, parents=["B"]),
    }
    payload = graph_payload(nodes, "B", view="family")
    by_id = {n["id"]: n for n in payload["nodes"]}
    assert by_id["D1"]["x"] > by_id["B"]["x"]


def test_large_tree_spreads_in_both_axes():
    """Busy depth levels use a grid so 100-node trees are not one vertical column."""
    path = Path(__file__).resolve().parents[1] / "fixtures" / "synth-100-state.json"
    store = LocalStore(path)
    payload = graph_payload(store.nodes, "SYNTH-LAB/base-model", view="family")
    xs = [n["x"] for n in payload["nodes"]]
    ys = [n["y"] for n in payload["nodes"]]
    assert max(xs) - min(xs) > 120
    assert max(ys) - min(ys) > 120
    rounded = {(round(x, -1), round(y, -1)) for x, y in zip(xs, ys)}
    assert len(rounded) >= 40


def test_merge_family_has_single_merge_node():
    nodes = {
        "B": _node("B", carbon=1.0),
        "D1": _node("D1", carbon=0.1, parents=["B"]),
        "D2": _node("D2", carbon=0.2, parents=["B"]),
        "M": _node("M", carbon=0.05, parents=["D1", "D2"]),
    }
    payload = graph_payload(nodes, "B", view="family")
    assert payload["n_models"] == 4
    assert payload["n_edges"] == 4
