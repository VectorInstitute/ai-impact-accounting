"""Tests for the impact lineage graph payload and figure."""

import math
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


def test_radial_layout_places_child_away_from_root():
    nodes = {
        "B": _node("B", carbon=1.0),
        "D1": _node("D1", carbon=0.1, parents=["B"]),
    }
    payload = graph_payload(nodes, "B", view="family")
    by_id = {n["id"]: n for n in payload["nodes"]}
    b = by_id["B"]
    d1 = by_id["D1"]
    assert math.hypot(d1["x"] - b["x"], d1["y"] - b["y"]) > 80.0


def test_chain_extends_outward_from_parent():
    """A lone child continues in the parent's outward direction (not sideways/up)."""
    nodes = {
        "A": _node("A", carbon=0.01),
        "B": _node("B", carbon=0.01, parents=["A"]),
        "C": _node("C", carbon=0.01, parents=["B"]),
    }
    payload = graph_payload(nodes, "A", view="all")
    by_id = {n["id"]: n for n in payload["nodes"]}
    a, b, c = by_id["A"], by_id["B"], by_id["C"]
    assert b["y"] > a["y"] + 50
    assert c["y"] > b["y"] + 50
    assert abs(b["x"] - a["x"]) < 40
    assert abs(c["x"] - b["x"]) < 40


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


def test_fanout_level_has_minimum_separation():
    """Many siblings from one parent spread on a ring around the root."""
    parent = "distilbert-base-uncased"
    nodes = {parent: _node(parent, carbon=0.01)}
    for i in range(10):
        mid = f"DIA-MVP/child-{i}"
        nodes[mid] = _node(mid, carbon=0.001, parents=[parent])
    payload = graph_payload(nodes, parent, view="all")
    by_id = {n["id"]: n for n in payload["nodes"]}
    root = by_id[parent]
    positions = [(by_id[f"DIA-MVP/child-{i}"]["x"], by_id[f"DIA-MVP/child-{i}"]["y"]) for i in range(10)]
    for i, (x1, y1) in enumerate(positions):
        for x2, y2 in positions[i + 1 :]:
            assert math.hypot(x1 - x2, y1 - y2) >= 80.0
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    assert max(xs) - min(xs) > 120.0
    assert max(ys) - min(ys) > 120.0
    for x, y in positions:
        assert math.hypot(x - root["x"], y - root["y"]) > 80.0


def test_disconnected_two_node_chains_have_component_gap():
    """Tiny vertical chains must not pack flush (width≈0) against neighbors."""
    nodes = {}
    for i in range(4):
        parent = f"org/p-{i}"
        child = f"org/c-{i}"
        nodes[parent] = _node(parent, carbon=0.01)
        nodes[child] = _node(child, carbon=0.001, parents=[parent])
    payload = graph_payload(nodes, "org/p-0", view="all")
    by_id = {n["id"]: n for n in payload["nodes"]}
    groups = [{f"org/p-{i}", f"org/c-{i}"} for i in range(4)]

    def min_cross_dist(ga: set[str], gb: set[str]) -> float:
        best = math.inf
        for a in ga:
            for b in gb:
                ax, ay = by_id[a]["x"], by_id[a]["y"]
                bx, by = by_id[b]["x"], by_id[b]["y"]
                best = min(best, math.hypot(ax - bx, ay - by))
        return best

    for i in range(4):
        for j in range(i + 1, 4):
            assert min_cross_dist(groups[i], groups[j]) >= 95.0


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
