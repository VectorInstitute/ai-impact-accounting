"""Unit tests for the HF dataset store and flat export."""

from ai_impact_accounting.hub.store import PARQUET_FILE, _encode_flat_export, flatten_nodes
from ai_impact_accounting.models import Interval, Node, Report


def test_flatten_nodes_includes_footprint_and_lineage():
    node = Node(
        model_id="DIA-MVP/demo-model",
        report=Report(
            energy=Interval(1.0, 2.0),
            carbon=Interval(0.5, 0.5),
            water=Interval(10.0, 20.0),
            quality={"energy": "measured", "carbon": "estimated-from-region", "water": "estimated-from-default-wue"},
            gpu="A100",
            gpu_count=1,
            gpu_hours=2.5,
            region="ca-on",
            tool="dia-track",
            method="lora",
        ),
        lineage=[{"model": "meta-llama/Llama-3.2-3B-Instruct", "relation": "lora"}],
        updated_at="2026-06-28T00:00:00+00:00",
    )
    rows = flatten_nodes({"DIA-MVP/demo-model": node})
    assert len(rows) == 1
    row = rows[0]
    assert row["model_id"] == "DIA-MVP/demo-model"
    assert row["has_report"] is True
    assert row["parent_models"] == "meta-llama/Llama-3.2-3B-Instruct"
    assert row["relation"] == "lora"
    assert row["carbon_kgco2eq_lo"] == 0.5
    assert row["energy_quality"] == "measured"


def test_flatten_nodes_without_report():
    node = Node(model_id="org/no-report", lineage=[{"model": "base/model", "relation": "finetune"}])
    row = flatten_nodes({"org/no-report": node})[0]
    assert row["has_report"] is False
    assert row["parent_models"] == "base/model"
    assert "carbon_kgco2eq_lo" not in row


def test_encode_flat_export_writes_parquet():
    rows = [{"model_id": "a/b", "has_report": False, "parent_models": "", "relation": "", "updated_at": None}]
    path, data = _encode_flat_export(rows)
    assert path == PARQUET_FILE
    assert data.startswith(b"PAR1")
