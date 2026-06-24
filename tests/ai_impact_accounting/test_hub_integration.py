"""Integration tests that touch the Hugging Face Hub.

Marked ``integration_test`` so the default ``pytest -m "not integration_test"``
run skips them. They also self-skip when ``HF_TOKEN`` is absent.
"""

import os

import pytest

from ai_impact_accounting import fetch_meta


pytestmark = pytest.mark.integration_test


@pytest.mark.skipif(not os.getenv("HF_TOKEN"), reason="needs HF_TOKEN")
def test_fetch_meta_reads_a_public_card():
    meta = fetch_meta("distilbert-base-uncased")
    assert isinstance(meta, dict)
