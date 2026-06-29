"""Tests for producer-side track() instrumentation (estimate fallback path)."""

import json
import sys
from pathlib import Path

import jsonschema
import pytest

from ai_impact_accounting import track
from ai_impact_accounting.producer.tracking import (
    _codecarbon_supported,
    _detect_ci,
    _detect_pue,
    _detect_wue,
    _tdp_for,
)


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "src" / "ai_impact_accounting" / "schema" / "dia_schema.json"


def _validate_report(rep: dict) -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(instance=rep, schema=schema)


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
    _validate_report(rep)


def test_tdp_sizing_is_hardware_aware():
    assert _tdp_for("apple-silicon-mps") == 40  # laptop, not a 400W GPU
    assert _tdp_for("A100") == 400
    assert _tdp_for("NVIDIA A40") == 300
    assert 65 <= _tdp_for("cpu-8core") <= 150  # CPU clamp


def test_codecarbon_disabled_on_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert _codecarbon_supported() is False


def test_codecarbon_disabled_via_env(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("CODECARBON_DISABLED", "1")
    assert _codecarbon_supported() is False


def test_env_vars_flow_into_report(monkeypatch):
    monkeypatch.setenv("DIA_CI", "0.25")
    monkeypatch.setenv("DIA_PUE", "1.3")
    monkeypatch.setenv("DIA_WUE", "2.0,3.5")
    monkeypatch.setenv("DIA_REGION", "ca-central-1")
    with track(base_model="test/base", relation="finetune") as t:
        pass
    rep = t.report_dict()["dia_report"]
    ctx = rep["context"]
    assert ctx["region"] == "ca-central-1"
    assert ctx["carbon_intensity"] == 0.25
    assert ctx["wue_l_per_kwh"] == [2.0, 3.5]
    assert t.pue == 1.3
    assert rep["footprint"]["water_liters"]["quality"] == "estimated-from-region"
    _validate_report(rep)


def test_dia_wue_scalar_parses_to_pair(monkeypatch):
    monkeypatch.setenv("DIA_WUE", "3.0")
    assert _detect_wue() == (3.0, 3.0)
    with track(base_model="test/base", relation="finetune") as t:
        pass
    assert t.report_dict()["dia_report"]["context"]["wue_l_per_kwh"] == [3.0, 3.0]


def test_dia_wue_pair_parsing(monkeypatch):
    monkeypatch.setenv("DIA_WUE", "1.8,4.0")
    assert _detect_wue() == (1.8, 4.0)


def test_carbon_quality_defaulted_when_ci_not_supplied():
    with track(base_model="test/base", relation="finetune", region="test") as t:
        pass
    rep = t.report_dict()["dia_report"]
    if rep["footprint"]["energy_kwh"]["quality"] == "measured":
        assert rep["footprint"]["carbon_kgco2eq"]["quality"] == "estimated-from-region"
    else:
        assert rep["footprint"]["carbon_kgco2eq"]["quality"] == "estimated-from-region"


def test_carbon_quality_measured_when_ci_supplied(monkeypatch):
    monkeypatch.setenv("DIA_CI", "0.30")
    with track(base_model="test/base", relation="finetune", region="test") as t:
        pass
    rep = t.report_dict()["dia_report"]
    if rep["footprint"]["energy_kwh"]["quality"] == "measured":
        assert rep["footprint"]["carbon_kgco2eq"]["quality"] == "measured"


def test_detect_ci_and_pue_defaults(monkeypatch):
    monkeypatch.delenv("DIA_CI", raising=False)
    monkeypatch.delenv("DIA_PUE", raising=False)
    assert _detect_ci() == 0.40
    assert _detect_pue() == 1.1


@pytest.mark.parametrize(
    ("env_val", "expected"),
    [
        ("0.55", 0.55),
        ("1.25", 1.25),
    ],
)
def test_detect_ci_and_pue_from_env(monkeypatch, env_val, expected):
    if "." in env_val and float(env_val) < 1:
        monkeypatch.setenv("DIA_CI", env_val)
        assert _detect_ci() == expected
    else:
        monkeypatch.setenv("DIA_PUE", env_val)
        assert _detect_pue() == expected
