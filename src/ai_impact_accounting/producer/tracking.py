"""Producer-side instrumentation.

Wrap your training loop with :class:`track`; on exit it measures energy/carbon
(CodeCarbon if available, else a hardware-TDP estimate), derives water from a WUE
range, and emits a ``dia_report`` block you can inject into a model card before
``push_to_hub``.

.. code-block:: python

    from ai_impact_accounting import track

    with track(base_model="meta-llama/Llama-3-8B", relation="qlora") as t:
        train(...)
    t.push("you/llama3-8b-mydataset")  # updates the card on the Hub

Design goals: auto-detect everything, stamp each auto-derived field with its
data-quality tier, and require the user to supply only ``base_model`` + ``relation``.
"""

from __future__ import annotations

import os
import re
import sys
import time
from contextlib import ContextDecorator
from typing import Any, Literal, Optional


WUE_DEFAULT = (1.8, 4.0)
CI_DEFAULT = 0.40
PUE_DEFAULT = 1.1
TDP_W = {"V100": 300, "A100": 400, "A100-80GB": 400, "H100": 700, "H100-80GB": 700, "H800": 350}
APPLE_PACKAGE_W = 40  # M-series sustained package power under ML load (rough)
CPU_W_PER_CORE = 6  # rough package draw per active core under load
CPU_W_MIN, CPU_W_MAX = 65, 150  # clamp: a CPU socket is not a 400W datacenter GPU


def _detect_cpu_cores() -> int:
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def _detect_gpu() -> tuple[str, int]:
    try:
        import pynvml

        pynvml.nvmlInit()
        name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        name = name.decode() if isinstance(name, bytes) else name
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return name, count
    except Exception:
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0), torch.cuda.device_count()
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "apple-silicon-mps", 1
        except Exception:
            pass
    # No accelerator found -> this is a CPU-only run. Record the core count so the
    # estimator can size power realistically instead of assuming a 400W GPU.
    return f"cpu-{_detect_cpu_cores()}core", 1


def _tdp_for(name: str) -> int:
    n = name.replace("-", "").replace(" ", "").lower()
    if "apple" in n or "mps" in n:  # laptop, not a 400W datacenter GPU
        return APPLE_PACKAGE_W
    if n.startswith("cpu"):  # CPU-only: size from detected cores
        m = re.search(r"(\d+)core", n)
        cores = int(m.group(1)) if m else 1
        return max(CPU_W_MIN, min(CPU_W_MAX, cores * CPU_W_PER_CORE))
    for k, v in TDP_W.items():
        if k.replace("-", "").lower() in n:
            return v
    return 400


def _detect_region() -> str:
    """Return the grid region from env vars, or ``"unknown"``."""
    return os.getenv("DIA_REGION") or os.getenv("AWS_REGION") or "unknown"


def _codecarbon_supported() -> bool:
    """Return whether CodeCarbon should run on this host.

    On macOS, CodeCarbon invokes ``sudo powermetrics``, which prompts for a
    password and usually returns zero on Apple Silicon. Skip it and use the TDP
    estimate path instead.
    """
    if sys.platform == "darwin":
        return False
    try:
        import codecarbon  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


class track(ContextDecorator):  # noqa: N801  (public API: `with track(...)`)
    """Context manager that measures a training run and emits a ``dia_report``.

    Parameters
    ----------
    base_model : str
        The parent model id this run derives from.
    relation : str, optional
        Relation to the base (``finetune``, ``lora``, ``qlora``, ``quantized``,
        ``merge``, ``distill``, ``fork``, ``adapter``). Defaults to ``"finetune"``.
    region : str, optional
        Grid region; auto-detected from the environment when omitted.
    wue : tuple of float, optional
        Water-usage-effectiveness range in L/kWh.
    carbon_intensity : float, optional
        Grid carbon intensity in kgCO2/kWh, used for the estimate fallback.
    pue : float, optional
        Power-usage effectiveness multiplier.
    """

    def __init__(
        self,
        base_model: str,
        relation: str = "finetune",
        region: Optional[str] = None,
        wue: tuple[float, float] = WUE_DEFAULT,
        carbon_intensity: float = CI_DEFAULT,
        pue: float = PUE_DEFAULT,
    ) -> None:
        """Configure the tracker and auto-detect hardware/region."""
        self.base_model = base_model
        self.relation = relation
        self.region = region or _detect_region()
        self.wue = tuple(wue)
        self.ci = carbon_intensity
        self.pue = pue
        self.gpu, self.gpu_count = _detect_gpu()
        self.is_cpu = self.gpu.lower().startswith("cpu")
        self._tracker: Optional[Any] = None
        self._t0: Optional[float] = None
        self.energy_kwh: Optional[float] = None
        self.carbon_kg: Optional[float] = None
        self.gpu_hours: float = 0.0
        self.water_l: tuple[float, float] = (0.0, 0.0)
        self.quality: dict[str, str] = {}

    def __enter__(self) -> "track":
        """Start the clock (and CodeCarbon when supported on this host)."""
        self._t0 = time.time()
        if _codecarbon_supported():
            try:
                from codecarbon import EmissionsTracker  # noqa: PLC0415

                self._tracker = EmissionsTracker(log_level="error", save_to_file=False)
                self._tracker.start()
            except Exception:
                self._tracker = None
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        """Stop measuring and finalize energy/carbon/water with quality tiers."""
        wall_h = (time.time() - (self._t0 or time.time())) / 3600.0
        gpu_hours = round(wall_h * (self.gpu_count or 1), 4)

        if self._tracker is not None:
            try:
                self.carbon_kg = float(self._tracker.stop() or 0.0)  # kgCO2
                data = self._tracker.final_emissions_data
                self.energy_kwh = float(getattr(data, "energy_consumed", 0.0) or 0.0)
                self.quality = {"energy": "measured", "carbon": "measured"}
            except Exception:
                self._tracker = None

        # CodeCarbon can't read power on Apple Silicon / many laptops and returns
        # ~0. Don't launder a zero as "measured" -- fall through to the estimate.
        if (self.energy_kwh or 0.0) <= 0.0:
            self._tracker = None

        if self._tracker is None:  # TDP estimate
            tdp = _tdp_for(self.gpu)
            # CPU per-core figure already reflects under-load draw; the 0.70
            # utilization factor only applies to a GPU's rated TDP.
            pavg = tdp if self.is_cpu else tdp * 0.70
            self.energy_kwh = gpu_hours * pavg * self.pue / 1000.0
            self.carbon_kg = self.energy_kwh * self.ci
            self.quality = {"energy": "estimated-from-hardware", "carbon": "estimated-from-region"}

        self.gpu_hours = gpu_hours
        energy = self.energy_kwh or 0.0
        self.water_l = (energy * self.wue[0], energy * self.wue[1])
        self.quality["water"] = "estimated-from-default-wue"
        return False

    # ---- output --------------------------------------------------------------
    def report_dict(self) -> dict:
        """Build the ``dia_report`` block as a plain dict.

        Returns
        -------
        dict
            A ``{"dia_version": ..., "dia_report": {...}}`` mapping ready to be
            merged into model-card front matter.
        """
        return {
            "dia_version": "0.1",
            "dia_report": {
                "scope": "incremental",
                "lineage": [{"model": self.base_model, "relation": self.relation}],
                "compute": {
                    "hardware": {"gpu": self.gpu, "count": self.gpu_count},
                    "duration_gpu_hours": self.gpu_hours,
                },
                "footprint": {
                    "energy_kwh": {"value": round(self.energy_kwh or 0.0, 4), "quality": self.quality["energy"]},
                    "carbon_kgco2eq": {"value": round(self.carbon_kg or 0.0, 4), "quality": self.quality["carbon"]},
                    "water_liters": {
                        "value": [round(self.water_l[0], 3), round(self.water_l[1], 3)],
                        "quality": self.quality["water"],
                    },
                },
                "context": {
                    "region": self.region,
                    "carbon_intensity": self.ci,
                    "wue_l_per_kwh": list(self.wue),
                },
                "tool": "codecarbon" if self.quality["energy"] == "measured" else "dia-track-estimate",
            },
        }

    def to_yaml(self) -> str:
        """Return the ``dia_report`` block serialized as YAML."""
        import yaml

        return yaml.safe_dump(self.report_dict(), sort_keys=False)

    def checklist_line(self) -> str:
        """Return a one-line human-readable summary of the measured run."""
        e, c = self.energy_kwh or 0.0, self.carbon_kg or 0.0
        w = self.water_l
        eq = self.quality.get("energy", "unavailable")
        tag = "MEASURED" if eq == "measured" else f"ESTIMATED — {eq}"
        return (
            f"[{tag}] Training: {self.gpu_count}x {self.gpu}, {self.gpu_hours} GPU-h, "
            f"{e:.3g} kWh, {c:.3g} kgCO2eq, {w[0]:.2g}-{w[1]:.2g} L water "
            f"({self.region}). Base: {self.base_model}. "
            f"Tool: {self.report_dict()['dia_report']['tool']}."
        )

    def write(self, card_path: str = "README.md") -> str:
        """Inject/merge the ``dia_report`` into a local model-card README.

        Parameters
        ----------
        card_path : str, optional
            Path to the local model-card file. Defaults to ``"README.md"``.

        Returns
        -------
        str
            The path that was written.
        """
        from huggingface_hub import ModelCard

        try:
            card = ModelCard.load(card_path)
        except Exception:
            card = ModelCard("---\n---\n")
        rep = self.report_dict()
        card.data["dia_version"] = rep["dia_version"]
        card.data["dia_report"] = rep["dia_report"]
        card.save(card_path)
        return card_path

    def push(self, repo_id: str, token: Optional[str] = None) -> str:
        """Update the model card on the Hub in place.

        Parameters
        ----------
        repo_id : str
            Target model repo id on the Hub.
        token : str, optional
            Hugging Face write token; falls back to ``HF_TOKEN``.

        Returns
        -------
        str
            The repo id that was updated.
        """
        from huggingface_hub import ModelCard

        token = token or os.getenv("HF_TOKEN")
        try:
            card = ModelCard.load(repo_id, token=token)
        except Exception:
            card = ModelCard("---\n---\n")
        rep = self.report_dict()
        card.data["dia_version"] = rep["dia_version"]
        card.data["dia_report"] = rep["dia_report"]
        card.push_to_hub(repo_id, token=token)
        return repo_id
