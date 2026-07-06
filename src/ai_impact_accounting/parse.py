"""Extract a normalized :class:`~ai_impact_accounting.models.Report` and lineage.

Lineage resolution order:

1. ``dia_report.lineage`` (explicit, with relation types).
2. Hugging Face native ``base_model`` + ``base_model_relation`` metadata (fallback).

Partial reports are accepted on purpose: a node may declare lineage with no
footprint (still useful for the DAG), or a footprint with no lineage (a root).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .models import (
    CI_IMPUTE_RANGE,
    METHOD_GPU_HOURS,
    PUE_DEFAULT,
    TDP_UTILIZATION,
    TDP_W,
    WUE_DEFAULT,
    Interval,
    Report,
)


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
    return r, []


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
    # energy = gpu_h * Pavg * PUE ; Pavg ~ TDP * utilization (kWh)
    tdp_kw = TDP_W["A100"] / 1000.0
    util = sum(TDP_UTILIZATION) / len(TDP_UTILIZATION)
    e = gpu_h * tdp_kw * util * PUE_DEFAULT
    r.energy = Interval(e, e)
    r.carbon = Interval(e * CI_IMPUTE_RANGE[0], e * CI_IMPUTE_RANGE[1])
    r.water = Interval(e * WUE_DEFAULT[0], e * WUE_DEFAULT[1])
    r.quality = {"energy": "imputed", "carbon": "imputed", "water": "imputed"}
    return r


_META_TABLE_ROW = re.compile(
    r"Llama 3 (8B|70B)\s*\|\s*([^|]+?)\|\s*([^|]+?)\|\s*([^|]+?)\|",
    re.IGNORECASE,
)
_META_HTML_ROW = re.compile(
    r"Llama 3 (8B|70B)\s*</td>\s*<td>\s*([^<]+?)\s*</td>\s*<td>\s*([^<]+?)\s*</td>\s*<td>\s*([^<]+?)\s*</td>",
    re.IGNORECASE | re.DOTALL,
)
_META_HARDWARE = re.compile(
    r"hardware of type\s+([^(|\n]+?)\s*\(\s*TDP of\s*(\d+)\s*W\s*\)",
    re.IGNORECASE,
)
_META_TOTAL_CARBON = re.compile(
    r"Estimated total emissions were\s+([\d,.]+)\s*tCO2eq",
    re.IGNORECASE,
)
_META_TOTAL_CARBON = re.compile(
    r"Estimated total emissions were\s+([\d,.]+)\s*tCO2eq",
    re.IGNORECASE,
)
_LLAMA_GHG_VARIANT = r"3\.\d+\s+(?:\d+B|\d+\.\d+B)(?:\s+(?:SpinQuant|QLora))?"
_META_GHG_HTML_ROW = re.compile(
    rf"Llama ({_LLAMA_GHG_VARIANT})\s*</td>\s*<td>\s*([^<]+?)\s*</td>\s*<td>\s*([^<]+?)\s*</td>\s*<td>\s*([^<]+?)\s*</td>\s*<td>\s*([^<]+?)\s*</td>",
    re.IGNORECASE | re.DOTALL,
)
_META_GHG_MD_ROW_6 = re.compile(
    rf"^\|\s*Llama ({_LLAMA_GHG_VARIANT})\s*\|\s*([^|\n]+)\|\s*([^|\n]+)\|\s*([^|\n]+)\|\s*([^|\n]+)\|\s*([^|\n]+)\|",
    re.IGNORECASE | re.MULTILINE,
)
_META_GHG_MD_ROW_5 = re.compile(
    rf"^\|\s*Llama ({_LLAMA_GHG_VARIANT})\s*\|\s*([^|\n]+)\|\s*([^|\n]+)\|\s*([^|\n]+)\|\s*([^|\n]+)\|",
    re.IGNORECASE | re.MULTILINE,
)
_META_GHG_TOTAL = re.compile(
    r"Estimated total location-based greenhouse gas emissions were\s+\*?\*?([\d,.]+)\*?\*?\s*tons?\s*CO2eq",
    re.IGNORECASE,
)
_H100_HARDWARE = re.compile(
    r"(H100-80GB)(?:\s*\(\s*TDP of\s*(\d+)\s*W\s*\))?",
    re.IGNORECASE,
)


def _parse_tonnes(cell: str) -> float | None:
    cleaned = re.sub(r"\*+", "", (cell or "")).strip()
    if not cleaned or "negligible" in cleaned.lower():
        return None
    raw = re.sub(r"[^\d.]", "", cleaned)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_magnitude(text: str) -> float | None:
    raw = (text or "").strip().replace(",", "").replace("\\", "")
    match = re.match(r"^([\d.]+)\s*([kKmM])?$", raw)
    if not match:
        return None
    val = float(match.group(1))
    suffix = (match.group(2) or "").upper()
    if suffix == "K":
        val *= 1_000
    elif suffix == "M":
        val *= 1_000_000
    return val


def _format_carbon_kg(tonnes: float, *, location_based: bool = True) -> str:
    """Format publisher tonnes as kgCO₂eq for dashboard display (DIA reports use kg)."""
    suffix = " (location-based)" if location_based else ""
    return f"{Interval._num(tonnes * 1000.0)} kgCO₂eq{suffix}"


def _card_has_carbon_section(card_text: str) -> bool:
    lower = card_text.lower()
    return any(
        phrase in lower
        for phrase in (
            "carbon footprint",
            "greenhouse gas emissions",
            "greenhouse gas emission",
        )
    )


def _meta_ghg_variant_hint(model_id: str) -> str | None:
    mid = model_id.lower()
    if "llama" not in mid:
        return None
    ver_m = re.search(r"llama[-_.]?3(?:\.(\d+))?", mid)
    if not ver_m:
        return None
    minor = ver_m.group(1)
    version = f"3.{minor}" if minor else "3"
    for pat, label in (
        (r"405b", "405B"),
        (r"70b", "70B"),
        (r"13b", "13B"),
        (r"8b", "8B"),
        (r"3b", "3B"),
        (r"1b", "1B"),
    ):
        if re.search(pat, mid):
            return f"Llama {version} {label}"
    return None


def _normalize_ghg_row_key(name_fragment: str) -> str:
    return re.sub(r"\s+", " ", f"Llama {name_fragment.strip()}")


def _meta_ghg_rows(card_text: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}

    def ingest(key: str, hours: str, power: str, carbon_loc: str, carbon_mkt: str) -> None:
        if key.lower() in {"total", "llama total"}:
            return
        tonnes = _parse_tonnes(carbon_loc)
        if tonnes is None:
            return
        rows[key] = {
            "variant": key,
            "gpu_hours": hours.strip().replace("\\", ""),
            "gpu_hours_value": _parse_magnitude(hours),
            "power_w": power.strip(),
            "carbon_tonnes": tonnes,
            "carbon_market_tonnes": _parse_tonnes(carbon_mkt),
        }

    for match in _META_GHG_HTML_ROW.finditer(card_text):
        name, hours, power, carbon_loc, carbon_mkt = match.groups()
        ingest(_normalize_ghg_row_key(name), hours, power, carbon_loc, carbon_mkt)

    for match in _META_GHG_MD_ROW_6.finditer(card_text):
        name, hours, _logit, power, carbon_loc, carbon_mkt = match.groups()
        ingest(_normalize_ghg_row_key(name), hours, power, carbon_loc, carbon_mkt)

    for match in _META_GHG_MD_ROW_5.finditer(card_text):
        name, hours, power, carbon_loc, carbon_mkt = match.groups()
        key = _normalize_ghg_row_key(name)
        if key not in rows:
            ingest(key, hours, power, carbon_loc, carbon_mkt)

    return rows


def _ghg_disclosure_result(
    row: dict[str, Any],
    *,
    hardware: str | None,
    card_power: str | None,
) -> dict[str, Any]:
    tonnes = row["carbon_tonnes"]
    market_tonnes = row.get("carbon_market_tonnes")
    market_kg = market_tonnes * 1000.0 if market_tonnes is not None else None
    return {
        "source": "model_card",
        "scope": "pretraining",
        "quality": "disclosed-on-card",
        "variant": row["variant"],
        "carbon": _format_carbon_kg(tonnes),
        "carbon_tonnes": tonnes,
        "carbon_kg": tonnes * 1000.0,
        "carbon_market_tonnes": market_tonnes,
        "carbon_market_kg": market_kg,
        "carbon_market": _format_carbon_kg(market_tonnes, location_based=False) if market_tonnes is not None else None,
        "gpu_hours": row["gpu_hours"],
        "gpu_hours_value": row["gpu_hours_value"],
        "power_w": row["power_w"] or card_power,
        "hardware": hardware,
        "notes": (
            "From Meta's training greenhouse-gas table on the model card (not a dia_report). "
            "Location-based pretraining total for this variant."
        ),
    }


def _parse_meta_ghg_disclosure(card_text: str, model_id: str) -> dict[str, Any] | None:
    """Meta Llama 3.1+ cards with Training Greenhouse Gas Emissions tables."""
    if "greenhouse gas" not in card_text.lower():
        return None

    rows = _meta_ghg_rows(card_text)
    if not rows:
        return None

    hw_match = _H100_HARDWARE.search(card_text)
    hardware = hw_match.group(1) if hw_match else None
    card_power = hw_match.group(2) if hw_match and hw_match.group(2) else None

    variant = _meta_ghg_variant_hint(model_id)
    row = rows.get(variant) if variant else None
    if row and row.get("carbon_tonnes"):
        return _ghg_disclosure_result(row, hardware=hardware, card_power=card_power)

    total = _META_GHG_TOTAL.search(card_text)
    if not total:
        return None
    tonnes = float(total.group(1).replace(",", ""))
    return {
        "source": "model_card",
        "scope": "pretraining",
        "quality": "disclosed-on-card",
        "carbon": _format_carbon_kg(tonnes),
        "carbon_tonnes": tonnes,
        "carbon_kg": tonnes * 1000.0,
        "hardware": hardware,
        "power_w": card_power,
        "notes": (
            "Parsed from model card prose (combined variant total). "
            "Use a size-specific repo id (e.g. 8B, 70B) for per-variant rows."
        ),
    }


def _parse_llama3_disclosure(card_text: str, model_id: str) -> dict[str, Any] | None:
    if "carbon footprint" not in card_text.lower():
        return None

    rows = _meta_carbon_rows(card_text)

    hw_match = _META_HARDWARE.search(card_text)
    hardware = hw_match.group(1).strip() if hw_match else None
    card_power = hw_match.group(2).strip() if hw_match else None

    size = _model_size_hint(model_id)
    if size and size in rows and rows[size]["carbon_tonnes"]:
        row = rows[size]
        tonnes = row["carbon_tonnes"]
        return {
            "source": "model_card",
            "scope": "pretraining",
            "quality": "disclosed-on-card",
            "variant": row["variant"],
            "carbon": _format_carbon_kg(tonnes, location_based=False),
            "carbon_tonnes": tonnes,
            "carbon_kg": tonnes * 1000.0,
            "gpu_hours": row["gpu_hours"],
            "gpu_hours_value": row["gpu_hours_value"],
            "power_w": row["power_w"] or card_power,
            "hardware": hardware,
            "notes": (
                "From the model card hardware/carbon table (not a dia_report). "
                "Pretraining total — not incremental DIA scope."
            ),
        }

    total = _META_TOTAL_CARBON.search(card_text)
    if not total:
        return None
    tonnes = float(total.group(1).replace(",", ""))
    return {
        "source": "model_card",
        "scope": "pretraining",
        "quality": "disclosed-on-card",
        "carbon": _format_carbon_kg(tonnes, location_based=False),
        "carbon_tonnes": tonnes,
        "carbon_kg": tonnes * 1000.0,
        "hardware": hardware,
        "power_w": card_power,
        "notes": (
            "Parsed from model card prose (combined variant total). "
            "Use an 8B- or 70B-specific repo id for per-variant table rows."
        ),
    }


def _model_size_hint(model_id: str) -> str | None:
    mid = model_id.lower()
    if "70b" in mid:
        return "70B"
    if "8b" in mid:
        return "8B"
    return None


def _meta_carbon_rows(card_text: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}

    def _add_row(variant: str, hours: str, power: str, carbon: str) -> None:
        tonnes_raw = re.sub(r"[^\d.]", "", carbon.strip())
        tonnes = float(tonnes_raw) if tonnes_raw else None
        key = variant.upper()
        rows[key] = {
            "variant": f"Llama 3 {key}",
            "gpu_hours": hours.strip(),
            "gpu_hours_value": _parse_magnitude(hours),
            "power_w": power.strip(),
            "carbon_tonnes": tonnes,
        }

    for match in _META_HTML_ROW.finditer(card_text):
        _add_row(*match.groups())
    for match in _META_TABLE_ROW.finditer(card_text):
        _add_row(*match.groups())
    return rows


def parse_card_disclosure(card_text: str, model_id: str) -> dict[str, Any] | None:
    """Extract published pretraining footprint from model card prose/tables.

    Supports Meta Llama 3 / 3.1 / 3.2 / 3.3 hardware/carbon sections when no ``dia_report`` exists.
    """
    if not card_text or not _card_has_carbon_section(card_text):
        return None

    ghg = _parse_meta_ghg_disclosure(card_text, model_id)
    if ghg:
        return ghg
    return _parse_llama3_disclosure(card_text, model_id)
