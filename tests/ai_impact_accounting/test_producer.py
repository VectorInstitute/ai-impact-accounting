"""Tests for producer-side track() instrumentation (estimate fallback path)."""

import sys

from ai_impact_accounting import track
from ai_impact_accounting.producer.tracking import _codecarbon_supported, _tdp_for


def test_track_emits_incremental_report_with_quality_tiers():
    with track(base_model="meta-llama/Llama-3-8B", relation="qlora", region="test") as t:
        pass  # no training; exercises the timing + estimate path
    rep = t.report_dict()["dia_report"]
    assert rep["scope"] == "incremental"
    assert rep["lineage"] == [{"model": "meta-llama/Llama-3-8B", "relation": "qlora"}]
    fp = rep["footprint"]
    assert set(fp) == {"energy_kwh", "carbon_kgco2eq", "water_liters"}
    # Water is always a default-WUE estimate; energy/carbon are measured or estimated.
    assert fp["water_liters"]["quality"] == "estimated-from-default-wue"
    assert t.quality["energy"] in ("measured", "estimated-from-hardware")


def test_tdp_sizing_is_hardware_aware():
    assert _tdp_for("apple-silicon-mps") == 40  # laptop, not a 400W GPU
    assert _tdp_for("A100") == 400
    assert 65 <= _tdp_for("cpu-8core") <= 150  # CPU clamp


def test_codecarbon_disabled_on_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert _codecarbon_supported() is False
