"""Tests for the ``dia`` CLI validation against the packaged schema."""

from ai_impact_accounting.producer.cli import _load_schema, cmd_validate


VALID_CARD = """---
dia_report:
  scope: incremental
  footprint:
    energy_kwh:
      value: 1.0
      quality: measured
    carbon_kgco2eq:
      value: 0.4
      quality: measured
---

Model card body.
"""

CUMULATIVE_CARD = VALID_CARD.replace("scope: incremental", "scope: cumulative")

NO_BLOCK_CARD = """---
license: mit
---

Body.
"""


def _write(tmp_path, text):
    p = tmp_path / "README.md"
    p.write_text(text)
    return str(p)


def test_schema_is_packaged_and_loadable():
    schema = _load_schema()
    assert schema["title"].startswith("DIA Report")


def test_validate_accepts_incremental(tmp_path):
    assert cmd_validate(_write(tmp_path, VALID_CARD)) == 0


def test_validate_rejects_cumulative(tmp_path):
    assert cmd_validate(_write(tmp_path, CUMULATIVE_CARD)) == 1


def test_validate_fails_without_block(tmp_path):
    assert cmd_validate(_write(tmp_path, NO_BLOCK_CARD)) == 1
