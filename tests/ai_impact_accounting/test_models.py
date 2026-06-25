"""Tests for interval math and report (de)serialization."""

from ai_impact_accounting import Interval, Report
from ai_impact_accounting.models import as_interval


def test_as_interval_scalar_pair_none():
    assert as_interval(None) == (0.0, 0.0)
    assert as_interval(5) == (5.0, 5.0)
    assert as_interval([2, 1]) == (1.0, 2.0)  # ordered lo<=hi


def test_interval_add_and_format():
    a = Interval(1.0, 2.0)
    b = Interval(3.0, 4.0)
    s = a + b
    assert (s.lo, s.hi) == (4.0, 6.0)
    assert Interval(5.0, 5.0).fmt(" kg") == "5.0 kg"
    assert "–" in Interval(1.0, 2.0).fmt(" L")  # en-dash range


def test_report_round_trip():
    r = Report(scope="incremental")
    r.energy = Interval(1.0, 1.0)
    r.carbon = Interval(0.4, 0.4)
    r.quality = {"carbon": "measured"}
    r2 = Report.from_dict(r.to_dict())
    assert r2.scope == "incremental"
    assert (r2.carbon.lo, r2.carbon.hi) == (0.4, 0.4)
    assert r2.quality["carbon"] == "measured"
