"""Synthetic dataset generator and large-graph smoke tests."""

import importlib.util
import json
from pathlib import Path

from ai_impact_accounting import LocalStore, rollup
from ai_impact_accounting.dashboard.api import dashboard_payload, graph_vis_payload


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts" / "generate_synthetic_state.py"


def _import_builder():
    spec = importlib.util.spec_from_file_location("generate_synthetic_state", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_build_synthetic_nodes_count():
    mod = _import_builder()
    nodes = mod.build_synthetic_nodes(100, seed=1)
    assert len(nodes) == 100
    assert "SYNTH-LAB/base-model" in nodes
    assert sum(1 for n in nodes.values() if n.report) >= 50


def test_synthetic_fixture_loads_and_dashboard_ok(tmp_path):
    mod = _import_builder()
    nodes = mod.build_synthetic_nodes(100, seed=7)
    path = tmp_path / "state.json"
    path.write_text(json.dumps(mod.serialize_nodes(nodes)), encoding="utf-8")

    store = LocalStore(path)
    assert len(store.nodes) == 100

    out = dashboard_payload(store, "SYNTH-LAB/base-model", graph_view="all")
    assert out["ok"] is True
    assert out["graph"]["n_models"] == 100
    assert out["graph"]["n_edges"] == 99

    graph = graph_vis_payload(store.nodes, "SYNTH-LAB/base-model", view="all")
    assert len(graph["nodes"]) == 100


def test_synthetic_family_rollup():
    mod = _import_builder()
    nodes = mod.build_synthetic_nodes(100, seed=3)
    res = rollup(nodes, "SYNTH-LAB/base-model")
    assert res["n_models"] == 100
    assert res["n_with_report"] >= 1
