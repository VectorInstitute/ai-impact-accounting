"""Data and Impact Accounting (DIA) for open-source AI footprint tracking.

This package tracks the carbon and water footprint of open-source AI models and
their derivatives across three planes:

- **producer** -- :class:`track` instruments a training run and stamps a
  ``dia_report`` block into the model card.
- **core** -- :func:`rollup` builds the lineage DAG and aggregates a family's
  footprint with coverage and per-field quality provenance.
- **hub** -- :class:`Store`, :func:`ingest_model`, and the crawler collect
  reports from the Hub (operator side).

The Gradio dashboard lives in :mod:`ai_impact_accounting.dashboard` and requires
the optional ``dashboard`` extra, so it is not imported here.
"""

from .graph import build_graph, family_members, rollup
from .hub import (
    Store,
    crawl_once,
    fetch_meta,
    ingest_model,
    list_derivatives,
    start_scheduler,
)
from .models import Interval, Node, Report
from .parse import impute_from_method, parse_lineage, parse_report
from .producer import track


__all__ = [
    "Interval",
    "Node",
    "Report",
    "Store",
    "build_graph",
    "crawl_once",
    "family_members",
    "fetch_meta",
    "impute_from_method",
    "ingest_model",
    "list_derivatives",
    "parse_lineage",
    "parse_report",
    "rollup",
    "start_scheduler",
    "track",
]
