"""Tests for card parsing, scope enforcement, lineage, and imputation."""

from ai_impact_accounting import impute_from_method, parse_lineage, parse_report
from ai_impact_accounting.models import Report
from ai_impact_accounting.parse import parse_card_disclosure


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


META_LLAMA_CARBON_SNIPPET = """
## Hardware and Software

**Carbon Footprint** Pretraining utilized a cumulative 7.7M GPU hours of computation on
hardware of type H100-80GB (TDP of 700W). Estimated total emissions were 2290 tCO2eq,
100% of which were offset by Meta's sustainability program.

| | **Time (GPU hours)** | **Power Consumption (W)** | **Carbon Emitted(tCO2eq)** |
| Llama 3 8B | 1.3M | 700 | 390 |
| Llama 3 70B | 6.4M | 700 | 1900 |
| Total | 7.7M | | 2290 |
"""


def test_parse_card_disclosure_llama33_70b():
    snippet = """
**Training Greenhouse Gas Emissions** Estimated total location-based greenhouse gas emissions were **11,390** tons CO2eq.
| Llama 3.3 70B | 7.0M | 700 | 2,040 | 0 |
"""
    out = parse_card_disclosure(snippet, "meta-llama/Llama-3.3-70B-Instruct")
    assert out is not None
    assert out["carbon_tonnes"] == 2040.0
    assert out["carbon_kg"] == 2_040_000.0
    assert out["carbon"] == "2,040,000 kgCO₂eq (location-based)"
    assert out["variant"] == "Llama 3.3 70B"
    assert out["gpu_hours"] == "7.0M"


def test_parse_card_disclosure_llama31_8b_html():
    snippet = """
**Training Greenhouse Gas Emissions** Estimated total location-based greenhouse gas emissions were **11,390** tons CO2eq.
<table>
  <tr><td>Llama 3.1 8B</td><td>1.46M</td><td>700</td><td>420</td><td>0</td></tr>
  <tr><td>Llama 3.1 70B</td><td>7.0M</td><td>700</td><td>2,040</td><td>0</td></tr>
</table>
"""
    out = parse_card_disclosure(snippet, "meta-llama/Llama-3.1-8B-Instruct")
    assert out is not None
    assert out["carbon_tonnes"] == 420.0
    assert out["variant"] == "Llama 3.1 8B"


def test_parse_card_disclosure_llama32_3b():
    snippet = """
**Training Greenhouse Gas Emissions:** Estimated total location-based greenhouse gas emissions were **240** tons CO2eq.
Training utilized a cumulative of 916k GPU hours of computation on H100-80GB (TDP of 700W) type hardware.

|  | Training Time (GPU hours) | Logit Generation Time (GPU Hours) | Training Power Consumption (W) | Training Location-Based Greenhouse Gas Emissions (tons CO2eq) | Training Market-Based Greenhouse Gas Emissions (tons CO2eq) |
| Llama 3.2 1B | 370k | - | 700 | 107 | 0 |
| Llama 3.2 3B | 460k | - | 700 | 133 | 0 |
| Total | 833k | 86k |  | 240 | 0 |
"""
    out = parse_card_disclosure(snippet, "meta-llama/Llama-3.2-3B-Instruct")
    assert out is not None
    assert out["carbon_tonnes"] == 133.0
    assert out["variant"] == "Llama 3.2 3B"
    assert out["gpu_hours"] == "460k"
    assert out["hardware"] == "H100-80GB"


def test_parse_card_disclosure_meta_llama_8b_html_table():
    html_snippet = """
**Carbon Footprint** Pretraining utilized hardware of type H100-80GB (TDP of 700W).
<table>
  <tr><td>Llama 3 8B</td><td>1.3M</td><td>700</td><td>390</td></tr>
  <tr><td>Llama 3 70B</td><td>6.4M</td><td>700</td><td>1900</td></tr>
</table>
"""
    out = parse_card_disclosure(html_snippet, "meta-llama/Meta-Llama-3-8B")
    assert out is not None
    assert out["carbon_tonnes"] == 390.0


def test_parse_card_disclosure_meta_llama_8b():
    out = parse_card_disclosure(META_LLAMA_CARBON_SNIPPET, "meta-llama/Meta-Llama-3-8B")
    assert out is not None
    assert out["carbon"] == "390,000 kgCO₂eq"
    assert out["carbon_tonnes"] == 390.0
    assert out["variant"] == "Llama 3 8B"
    assert out["gpu_hours"] == "1.3M"
    assert out["hardware"] == "H100-80GB"
    assert out["scope"] == "pretraining"


def test_parse_card_disclosure_meta_llama_70b():
    out = parse_card_disclosure(META_LLAMA_CARBON_SNIPPET, "meta-llama/Meta-Llama-3-70B")
    assert out is not None
    assert out["carbon_tonnes"] == 1900.0
