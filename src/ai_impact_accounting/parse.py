"""Extract a normalized :class:`~ai_impact_accounting.models.Report` and lineage.

Lineage resolution order:

1. ``dia_report.lineage`` (explicit, with relation types).
2. Hugging Face native ``base_model`` + ``base_model_relation`` metadata (fallback).

Partial reports are accepted on purpose: a node may declare lineage with no
footprint (still useful for the DAG), or a footprint with no lineage (a root).
"""

from __future__ import annotations

from typing import Any, Optional

from .models import METHOD_GPU_HOURS, Interval, Report


# HF base_model_relation -> our relation vocabulary
_HF_REL = {
    "finetune": "finetune",
    "adapter": "adapter",
    "lora": "lora",
    "quantized": "quantized",
    "merge": "merge",
    "fork": "fork",
}


def _quality(metric: Any, default: str) -> str:
    if isinstance(metric, dict):
        return metric.get("quality", default)
    return default


def _value(metric: Any) -> Any:
    return metric.get("value") if isinstance(metric, dict) else metric


def parse_lineage(meta: dict) -> list[dict]:
    """Resolve parent edges from card metadata.

    Parameters
    ----------
    meta : dict
        Model-card front-matter metadata.

    Returns
    -------
    list of dict
        De-duplicated ``{"model": ..., "relation": ...}`` entries, explicit
        ``dia_report.lineage`` first, then the HF native ``base_model`` fallback.
    """
    dia = meta.get("dia_report") or {}
    lineage = []
    seen = set()

    for item in dia.get("lineage", []) or []:
        if isinstance(item, dict) and item.get("model"):
            mid = item["model"]
            if mid not in seen:
                lineage.append({"model": mid, "relation": item.get("relation", "finetune")})
                seen.add(mid)

    # fallback: HF native base_model field (string or list)
    base = meta.get("base_model")
    rel = meta.get("base_model_relation", "finetune")
    bases = [base] if isinstance(base, str) else (base or [])
    for b in bases:
        if b and b not in seen:
            lineage.append({"model": b, "relation": _HF_REL.get(rel, "finetune")})
            seen.add(b)
    return lineage


def parse_report(meta: dict) -> tuple[Optional[Report], list[str]]:
    """Parse the ``dia_report`` block into a normalized report.

    Parameters
    ----------
    meta : dict
        Model-card front-matter metadata.

    Returns
    -------
    tuple of (Report or None, list of str)
        The parsed report (``None`` if there is no usable ``dia_report`` block,
        or it declares a non-incremental scope) and a list of error messages.

    Notes
    -----
    A non-incremental ``scope`` is hard-rejected: cumulative reporting would
    double-count shared ancestors when the subtree is summed.
    """
    dia = meta.get("dia_report")
    if not isinstance(dia, dict):
        return None, []

    errors: list[str] = []
    scope = dia.get("scope", "incremental")
    if scope != "incremental":
        # Hard reject: cumulative scope breaks subtree summation.
        return None, [f"scope must be 'incremental', got '{scope}' — node ignored"]

    fp = dia.get("footprint", {}) or {}
    r = Report(scope="incremental")
    r.energy = Interval.of(_value(fp.get("energy_kwh")))
    r.carbon = Interval.of(_value(fp.get("carbon_kgco2eq")))
    r.water = Interval.of(_value(fp.get("water_liters")))
    r.quality = {
        "energy": _quality(fp.get("energy_kwh"), "unavailable"),
        "carbon": _quality(fp.get("carbon_kgco2eq"), "unavailable"),
        "water": _quality(fp.get("water_liters"), "unavailable"),
    }

    comp = dia.get("compute", {}) or {}
    hw = comp.get("hardware", {}) or {}
    r.gpu = hw.get("gpu")
    r.gpu_count = hw.get("count")
    r.gpu_hours = comp.get("duration_gpu_hours")
    ctx = dia.get("context", {}) or {}
    r.region = ctx.get("region")
    r.tool = dia.get("tool")

    # method = relation of the first lineage entry, if any (for imputation priors)
    lin = parse_lineage(meta)
    r.method = lin[0]["relation"] if lin else None

    # If footprint is entirely empty but we know the method, leave as zeros but
    # mark quality unavailable; the aggregator can choose to impute.
    return r, errors


def impute_from_method(r: Report) -> Report:
    """Fill a zero footprint from a compute prior, labelled ``imputed``.

    Parameters
    ----------
    r : Report
        A report whose energy is zero. Its :attr:`~Report.method` (and
        :attr:`~Report.gpu_hours` if present) drives the prior.

    Returns
    -------
    Report
        The same report with energy/carbon/water filled in and every footprint
        field's quality stamped ``"imputed"``. Returned unchanged if energy is
        already positive.
    """
    if r.energy.hi > 0:
        return r
    gpu_h = r.gpu_hours or METHOD_GPU_HOURS.get(r.method or "finetune", 100.0)
    # energy = gpu_h * Pavg * PUE ; Pavg ~ 400W*0.7, PUE 1.1  -> kWh
    e = gpu_h * 0.400 * 0.70 * 1.1
    r.energy = Interval(e, e)
    r.carbon = Interval(e * 0.10, e * 0.60)  # CI range
    r.water = Interval(e * 1.8, e * 4.0)  # WUE range
    r.quality = {"energy": "imputed", "carbon": "imputed", "water": "imputed"}
    return r
