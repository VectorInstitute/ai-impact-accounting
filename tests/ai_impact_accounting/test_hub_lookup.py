"""Tests for Hub lookup / ingest helpers."""

from unittest.mock import patch

from ai_impact_accounting import Node, Report
from ai_impact_accounting.dashboard.api import hub_ingest, hub_lookup
from ai_impact_accounting.models import Interval


class _FakeStore:
    def __init__(self, nodes: dict | None = None, token: str | None = "tok") -> None:
        self.nodes = nodes or {}
        self.repo = "test/dia-state"
        self.token = token


def _node(mid, carbon=1.0, parents=()):
    rep = Report(scope="incremental")
    rep.carbon = Interval(carbon, carbon)
    rep.water = Interval(0.1, 0.1)
    rep.energy = Interval(0.2, 0.2)
    rep.quality = {"carbon": "measured"}
    rep.method = "finetune"
    lineage = [{"model": p, "relation": "finetune"} for p in parents]
    return Node(model_id=mid, report=rep, lineage=lineage)


def test_hub_lookup_requires_model_id():
    out = hub_lookup(_FakeStore(), "")
    assert out["ok"] is False


def test_hub_lookup_rejects_scratch():
    out = hub_lookup(_FakeStore(), "scratch")
    assert out["ok"] is False
    assert out["model_url"] is None
    assert "from scratch" in out["message"].lower()


@patch("ai_impact_accounting.dashboard.api.fetch_model_card")
def test_hub_lookup_finds_report_on_hub(mock_fetch):
    mock_fetch.return_value = (
        {
            "dia_report": {
                "scope": "incremental",
                "footprint": {"carbon_kgco2eq": {"value": [1.0, 2.0], "quality": "measured"}},
            }
        },
        "",
    )
    store = _FakeStore()
    out = hub_lookup(store, "org/base-model")
    assert out["ok"] is True
    assert out["hub_has_report"] is True
    assert out["hub_footprint"]["carbon"] == "1.0–2.0 kgCO₂eq"
    assert out["can_ingest"] is True


@patch("ai_impact_accounting.dashboard.api.fetch_model_card")
def test_hub_lookup_shows_store_copy(mock_fetch):
    mock_fetch.return_value = ({"dia_report": {"scope": "incremental"}}, "")
    store = _FakeStore({"B": _node("B")})
    out = hub_lookup(store, "B")
    assert out["in_store"] is True
    assert out["in_store_has_report"] is True
    assert out["store_footprint"]["carbon"] == "1.0 kgCO₂eq"


def test_hub_ingest_requires_token():
    out = hub_ingest(_FakeStore(token=None), "org/model")
    assert out["ok"] is False
    assert "token" in out["error"].lower()


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


@patch("ai_impact_accounting.dashboard.api.fetch_model_card")
def test_hub_lookup_parses_card_disclosure(mock_fetch):
    mock_fetch.return_value = ({"license": "llama3"}, META_LLAMA_CARBON_SNIPPET)
    out = hub_lookup(_FakeStore(), "meta-llama/Meta-Llama-3-8B")
    assert out["ok"] is True
    assert out["hub_has_report"] is False
    assert out["card_disclosure"]["carbon_tonnes"] == 390.0


@patch("ai_impact_accounting.dashboard.api.ingest_model")
def test_hub_ingest_delegates(mock_ingest):
    mock_ingest.return_value = {"ok": True, "has_report": True, "lineage": [], "errors": []}
    store = _FakeStore()
    out = hub_ingest(store, "org/model")
    assert out["ok"] is True
    mock_ingest.assert_called_once_with("org/model", store, token="tok", persist=True)
