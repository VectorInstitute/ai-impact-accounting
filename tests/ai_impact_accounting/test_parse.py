"""Tests for card parsing, scope enforcement, lineage, and imputation."""

from ai_impact_accounting import impute_from_method, parse_lineage, parse_report
from ai_impact_accounting.models import Report


def _meta(scope="incremental", energy_q="measured"):
    return {
        "dia_report": {
            "scope": scope,
            "lineage": [{"model": "base/x", "relation": "lora"}],
            "footprint": {
                "energy_kwh": {"value": 2.0, "quality": energy_q},
                "carbon_kgco2eq": {"value": 0.8, "quality": "measured"},
                "water_liters": {"value": [3.6, 8.0], "quality": "estimated-from-default-wue"},
            },
        }
    }


def test_incremental_report_parses():
    report, errors = parse_report(_meta())
    assert errors == []
    assert report is not None
    assert report.scope == "incremental"
    assert (report.carbon.lo, report.carbon.hi) == (0.8, 0.8)
    assert report.method == "lora"  # from first lineage relation


def test_cumulative_scope_is_rejected():
    report, errors = parse_report(_meta(scope="cumulative"))
    assert report is None
    assert errors and "incremental" in errors[0]


def test_no_dia_block_returns_none():
    report, errors = parse_report({"license": "mit"})
    assert report is None
    assert errors == []


def test_lineage_explicit_then_hf_fallback_dedups():
    meta = {
        "dia_report": {"scope": "incremental", "lineage": [{"model": "a/b", "relation": "merge"}]},
        "base_model": ["a/b", "c/d"],  # a/b duplicates the explicit edge
        "base_model_relation": "finetune",
    }
    lineage = parse_lineage(meta)
    models = [edge["model"] for edge in lineage]
    assert models == ["a/b", "c/d"]  # a/b only once
    assert lineage[0]["relation"] == "merge"  # explicit relation wins


def test_impute_only_fills_zero_and_labels_imputed():
    measured = Report(method="lora")
    measured.energy.hi = 5.0
    assert impute_from_method(measured) is measured  # untouched

    empty = Report(method="lora")
    filled = impute_from_method(empty)
    assert filled.energy.hi > 0
    assert filled.quality == {"energy": "imputed", "carbon": "imputed", "water": "imputed"}
