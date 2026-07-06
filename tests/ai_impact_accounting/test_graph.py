"""Tests for the lineage DAG rollup: incremental subtree sum, DAG dedup, coverage."""

from ai_impact_accounting import Node, Report, rollup
from ai_impact_accounting.graph import build_graph
from ai_impact_accounting.models import Interval


def _node(mid, carbon=None, parents=(), quality="measured"):
    rep = None
    if carbon is not None:
        rep = Report(scope="incremental")
        rep.carbon = Interval(carbon, carbon)
        rep.energy = Interval(carbon, carbon)
        rep.quality = {"carbon": quality}
    lineage = [{"model": p, "relation": "finetune"} for p in parents]
    return Node(model_id=mid, report=rep, lineage=lineage)


def _family():
    # B -> D1, B -> D2, (D1, D2) -> M  (M is a merge with two parents)
    return {
        "B": _node("B", carbon=100.0),
        "D1": _node("D1", carbon=10.0, parents=["B"]),
        "D2": _node("D2", carbon=20.0, parents=["B"]),
        "M": _node("M", carbon=5.0, parents=["D1", "D2"]),
    }


def test_merge_node_counted_once():
    res = rollup(_family(), "B")
    assert res["n_models"] == 4
    assert res["base_footprint"]["carbon"].hi == 100.0
    # 10 + 20 + 5 = 35 (M counted ONCE despite two parents)
    assert res["deriv_footprint"]["carbon"].hi == 35.0
    assert res["total_footprint"]["carbon"].hi == 135.0


def test_coverage_and_ratio():
    res = rollup(_family(), "B")
    assert res["coverage"] == 1.0
    assert res["n_with_report"] == 4
    lo, hi = res["deriv_over_base_ratio"]
    assert round(lo, 2) == 0.35 and round(hi, 2) == 0.35


def test_carbon_by_quality_segregates_provenance():
    nodes = {
        "B": _node("B", carbon=100.0, quality="measured"),
        "D1": _node("D1", carbon=10.0, parents=["B"], quality="estimated-from-region"),
    }
    res = rollup(nodes, "B")
    cbq = res["carbon_by_quality"]
    assert cbq["measured"].hi == 100.0
    assert cbq["estimated"].hi == 10.0


def test_missing_node_lowers_coverage_without_impute():
    nodes = {
        "B": _node("B", carbon=100.0),
        "D1": _node("D1", carbon=None, parents=["B"]),  # no report
    }
    res = rollup(nodes, "B", impute=False)
    assert res["n_with_report"] == 1
    assert res["n_without_report"] == 1
    assert res["coverage"] == 0.5
    assert res["deriv_footprint"]["carbon"].hi == 0.0  # missing contributes nothing


def test_impute_fills_missing_and_labels_imputed():
    nodes = {
        "B": _node("B", carbon=100.0),
        "D1": _node("D1", carbon=None, parents=["B"]),
    }
    res = rollup(nodes, "B", impute=True)
    assert res["deriv_footprint"]["carbon"].hi > 0  # imputed prior added
    assert "imputed" in res["carbon_by_quality"]


def test_build_graph_keeps_scratch_placeholder():
    nodes = {
        "root-model": _node("root-model", carbon=1.0, parents=["scratch"]),
        "child": _node("child", carbon=2.0, parents=["root-model"]),
    }
    g = build_graph(nodes)
    assert "scratch" in g.nodes
    assert g.has_edge("scratch", "root-model")
    assert g.has_edge("root-model", "child")
    assert g.in_degree("scratch") == 0
