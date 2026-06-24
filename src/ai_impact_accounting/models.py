"""Internal data model for DIA accounting.

Everything is normalized to ``[low, high]`` intervals so that measured points and
estimated ranges add up consistently across a model family. The constants and
priors below come from the DIA paper (Table A1/A2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Union


# ---- defaults / constants (from the DIA paper, Table A1/A2) -------------------
WUE_DEFAULT = (1.8, 4.0)  # L/kWh, combined on-site + off-site
CI_DEFAULT = 0.40  # kgCO2/kWh, generic grid
PUE_DEFAULT = 1.1  # hyperscale
TDP_W = {  # vendor TDP upper bounds
    "V100": 300,
    "A100": 400,
    "A100-80GB": 400,
    "H100": 700,
    "H100-80GB": 700,
    "H800": 350,
}
TDP_UTILIZATION = (0.60, 0.80)  # actual draw as fraction of TDP

# Compute priors for imputing models that declare a method but no footprint.
# GPU-hours, used only when impute=True and clearly labelled as quality="imputed".
METHOD_GPU_HOURS = {
    "lora": 1.0,
    "qlora": 1.5,
    "adapter": 1.0,
    "quantized": 0.5,
    "merge": 0.2,
    "distill": 200.0,
    "finetune": 100.0,
    "fork": 0.0,
}

# A scalar value, a [lo, hi] pair, or nothing.
MetricValue = Optional[Union[float, Sequence[float]]]


def as_interval(v: MetricValue) -> tuple[float, float]:
    """Normalize a raw metric value to a ``(lo, hi)`` tuple.

    Parameters
    ----------
    v : float or sequence of float or None
        A scalar (becomes ``(v, v)``), a two-element ``[lo, hi]`` sequence, or
        ``None`` (becomes ``(0.0, 0.0)``).

    Returns
    -------
    tuple of float
        The value as an ordered ``(lo, hi)`` pair.
    """
    if v is None:
        return (0.0, 0.0)
    if isinstance(v, (int, float)):
        return (float(v), float(v))
    lo, hi = float(v[0]), float(v[-1])
    return (min(lo, hi), max(lo, hi))


@dataclass
class Interval:
    """A ``[lo, hi]`` interval that adds component-wise and formats for display."""

    lo: float = 0.0
    hi: float = 0.0

    @staticmethod
    def of(v: MetricValue) -> "Interval":
        """Build an :class:`Interval` from a scalar, a pair, or ``None``.

        Parameters
        ----------
        v : float or sequence of float or None
            Raw metric value; see :func:`as_interval`.

        Returns
        -------
        Interval
            The normalized interval.
        """
        lo, hi = as_interval(v)
        return Interval(lo, hi)

    def __add__(self, other: "Interval") -> "Interval":
        """Add two intervals endpoint-wise."""
        return Interval(self.lo + other.lo, self.hi + other.hi)

    @staticmethod
    def _num(x: float) -> str:
        if x == 0:
            return "0"
        if x >= 100:
            return f"{x:,.0f}"
        if x >= 1:
            return f"{x:,.1f}"
        return f"{x:.3g}"

    def fmt(self, unit: str = "") -> str:
        """Render the interval as a human-readable string.

        Parameters
        ----------
        unit : str, optional
            Unit suffix appended to the number(s), e.g. ``" kWh"``.

        Returns
        -------
        str
            A single number when ``lo == hi``, otherwise ``"lo-hi"``.
        """
        if abs(self.lo - self.hi) < 1e-9:
            return f"{self._num(self.lo)}{unit}"
        return f"{self._num(self.lo)}–{self._num(self.hi)}{unit}"


@dataclass
class Report:
    """Normalized, incremental footprint for ONE model node.

    A node reports only its own delta (``scope == "incremental"``); the family
    total is the subtree sum. Each footprint field carries a per-field quality
    tier in :attr:`quality` so measured and estimated numbers stay separable.
    """

    scope: str = "incremental"
    energy: Interval = field(default_factory=Interval)
    carbon: Interval = field(default_factory=Interval)
    water: Interval = field(default_factory=Interval)
    quality: dict[str, str] = field(default_factory=dict)
    method: Optional[str] = None
    gpu: Optional[str] = None
    gpu_count: Optional[int] = None
    gpu_hours: Optional[float] = None
    region: Optional[str] = None
    tool: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize the report to a plain JSON-friendly dict.

        Returns
        -------
        dict
            Footprint intervals as ``[lo, hi]`` lists plus metadata fields.
        """
        return {
            "scope": self.scope,
            "energy_kwh": [self.energy.lo, self.energy.hi],
            "carbon_kgco2eq": [self.carbon.lo, self.carbon.hi],
            "water_liters": [self.water.lo, self.water.hi],
            "quality": self.quality,
            "method": self.method,
            "gpu": self.gpu,
            "gpu_count": self.gpu_count,
            "gpu_hours": self.gpu_hours,
            "region": self.region,
            "tool": self.tool,
        }

    @staticmethod
    def from_dict(d: dict) -> "Report":
        """Reconstruct a :class:`Report` from its serialized form.

        Parameters
        ----------
        d : dict
            A dict produced by :meth:`to_dict`.

        Returns
        -------
        Report
            The reconstructed report.
        """
        r = Report(scope=d.get("scope", "incremental"))
        r.energy = Interval.of(d.get("energy_kwh"))
        r.carbon = Interval.of(d.get("carbon_kgco2eq"))
        r.water = Interval.of(d.get("water_liters"))
        r.quality = dict(d.get("quality", {}))
        for k in ("method", "gpu", "gpu_count", "gpu_hours", "region", "tool"):
            setattr(r, k, d.get(k))
        return r


@dataclass
class Node:
    """One model in the lineage graph, with its report and parent edges."""

    model_id: str
    report: Optional[Report] = None  # None = no DIA data disclosed
    lineage: list[dict[str, Any]] = field(default_factory=list)
    updated_at: Optional[str] = None

    @property
    def has_report(self) -> bool:
        """bool: Whether this node disclosed a DIA report."""
        return self.report is not None
