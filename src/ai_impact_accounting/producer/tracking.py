"""Producer-side instrumentation.

Wrap your training loop with :class:`track`; on exit it measures energy/carbon
(CodeCarbon if available, else NVML power sampling, else a hardware-TDP estimate),
derives water from a WUE range, and emits a ``dia_report`` block you can inject
into a model card before ``push_to_hub``.

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
import threading
import time
from contextlib import ContextDecorator
from typing import Any, Literal, Optional

from ai_impact_accounting.models import CI_DEFAULT, PUE_DEFAULT, TDP_W, WUE_DEFAULT


APPLE_PACKAGE_W = 40  # M-series sustained package power under ML load (rough)
CPU_W_PER_CORE = 6  # rough package draw per active core under load
CPU_W_MIN, CPU_W_MAX = 65, 150  # clamp: a CPU socket is not a 400W datacenter GPU
GPU_TDP_UTILIZATION = 0.70  # fraction of rated TDP assumed under load (TDP-estimate tier only)
NVML_SAMPLE_INTERVAL_S = 1.0


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


def _nvml_power_limit_w() -> Optional[int]:
    """Return the NVML power-management cap for GPU 0 in watts, or ``None``."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        pynvml.nvmlShutdown()
        return int(limit_mw // 1000)
    except Exception:
        return None


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
    limit_w = _nvml_power_limit_w()
    if limit_w is not None:
        return limit_w
    return 400


def _detect_region() -> str:
    """Return the grid region from env vars, or ``"unknown"``."""
    return os.getenv("DIA_REGION") or os.getenv("AWS_REGION") or "unknown"


def _detect_ci() -> float:
    """Return grid carbon intensity from env, else the generic default.

    TODO: live grid-intensity API lookup and a region->CI table keyed on
    :func:`_detect_region`.
    """
    raw = os.getenv("DIA_CI")
    if raw is not None:
        return float(raw)
    return CI_DEFAULT


def _detect_pue() -> float:
    """Return PUE from env, else the hyperscale default."""
    raw = os.getenv("DIA_PUE")
    if raw is not None:
        return float(raw)
    return PUE_DEFAULT


def _detect_wue() -> tuple[float, float]:
    """Return WUE range from env (``"lo,hi"`` or scalar), else the paper default."""
    raw = os.getenv("DIA_WUE")
    if raw is not None:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) >= 2:
            return (float(parts[0]), float(parts[1]))
        val = float(parts[0])
        return (val, val)
    return WUE_DEFAULT


def _gpu_utilization_factor() -> float:
    """Return the TDP-estimate utilization factor, overridable via ``DIA_GPU_UTIL``."""
    raw = os.getenv("DIA_GPU_UTIL")
    if raw is not None:
        return float(raw)
    return GPU_TDP_UTILIZATION


def _codecarbon_supported() -> bool:
    """Return whether CodeCarbon should run on this host.

    On macOS, CodeCarbon invokes ``sudo powermetrics``, which prompts for a
    password and usually returns zero on Apple Silicon. Skip it and use the TDP
    estimate path instead.

    Set ``CODECARBON_DISABLED=1`` to force the NVML / TDP fallback (e.g. when
    verifying the NVML measured path).
    """
    if os.getenv("CODECARBON_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    if sys.platform == "darwin":
        return False
    try:
        import codecarbon  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


class _NvmlPowerSampler:
    """Background sampler that integrates NVML power readings across all GPUs."""

    def __init__(self, interval_s: float = NVML_SAMPLE_INTERVAL_S) -> None:
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: list[tuple[float, list[float]]] = []
        self._nvml: Any = None
        self._handles: list[Any] = []

    def start(self) -> bool:
        """Start sampling if CUDA and NVML are available."""
        try:
            import pynvml
            import torch

            if not torch.cuda.is_available():
                return False
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count <= 0:
                pynvml.nvmlShutdown()
                return False
            self._nvml = pynvml
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True
        except Exception:
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                powers = [self._nvml.nvmlDeviceGetPowerUsage(h) / 1000.0 for h in self._handles]
                self._samples.append((time.time(), powers))
            except Exception:
                pass
            self._stop.wait(self._interval_s)

    def stop(self) -> float:
        """Stop sampling and return integrated device energy in kWh (before PUE)."""
        if self._thread is None:
            return 0.0
        self._stop.set()
        self._thread.join(timeout=self._interval_s + 2.0)
        try:
            if self._nvml is not None:
                self._nvml.nvmlShutdown()
        except Exception:
            pass
        if len(self._samples) < 2:
            return 0.0
        total_ws = 0.0
        for i in range(1, len(self._samples)):
            t0, p0 = self._samples[i - 1]
            t1, p1 = self._samples[i]
            dt = t1 - t0
            total_w = sum((a + b) / 2.0 for a, b in zip(p0, p1, strict=True))
            total_ws += total_w * dt
        return total_ws / 3_600_000.0  # W·s -> kWh


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
        wue: Optional[tuple[float, float]] = None,
        carbon_intensity: Optional[float] = None,
        pue: Optional[float] = None,
    ) -> None:
        """Configure the tracker and auto-detect hardware/region."""
        self.base_model = base_model
        self.relation = relation
        self.region = region or _detect_region()
        self._ci_user_supplied = carbon_intensity is not None or os.getenv("DIA_CI") is not None
        self._wue_user_supplied = wue is not None or os.getenv("DIA_WUE") is not None
        self.ci = carbon_intensity if carbon_intensity is not None else _detect_ci()
        self.pue = pue if pue is not None else _detect_pue()
        self.wue: tuple[float, float] = wue if wue is not None else _detect_wue()
        self.gpu, self.gpu_count = _detect_gpu()
        self.is_cpu = self.gpu.lower().startswith("cpu")
        self._tracker: Optional[Any] = None
        self._nvml_sampler: Optional[_NvmlPowerSampler] = None
        self._t0: Optional[float] = None
        self.energy_kwh: Optional[float] = None
        self.carbon_kg: Optional[float] = None
        self.gpu_hours: float = 0.0
        self.water_l: tuple[float, float] = (0.0, 0.0)
        self.quality: dict[str, str] = {}
        self._tool = "dia-track-estimate"

    def __enter__(self) -> "track":
        """Start the clock, plus CodeCarbon and NVML sampling when available."""
        self._t0 = time.time()
        if _codecarbon_supported():
            try:
                from codecarbon import EmissionsTracker  # noqa: PLC0415

                self._tracker = EmissionsTracker(log_level="error", save_to_file=False)
                self._tracker.start()
            except Exception:
                self._tracker = None
        # Always run the NVML sampler so we can fall back to it if CodeCarbon
        # measures nothing (returns ~0), rather than jumping to the TDP estimate.
        sampler = _NvmlPowerSampler()
        if sampler.start():
            self._nvml_sampler = sampler
        return self

    def _carbon_quality_for_measured_energy(self) -> str:
        """Carbon tier when device energy is measured but carbon is derived from CI."""
        return "measured" if self._ci_user_supplied else "estimated-from-region"

    def __exit__(self, *exc: object) -> Literal[False]:
        """Stop measuring and finalize energy/carbon/water with quality tiers."""
        wall_h = (time.time() - (self._t0 or time.time())) / 3600.0
        gpu_hours = round(wall_h * (self.gpu_count or 1), 4)

        # Always stop the NVML sampler (so its thread never leaks) and keep its
        # integrated energy as a fallback in case CodeCarbon measured nothing.
        nvml_kwh = 0.0
        if self._nvml_sampler is not None:
            try:
                nvml_kwh = self._nvml_sampler.stop()
            except Exception:
                nvml_kwh = 0.0

        if self._tracker is not None:
            try:
                self.carbon_kg = float(self._tracker.stop() or 0.0)  # kgCO2
                data = self._tracker.final_emissions_data
                self.energy_kwh = float(getattr(data, "energy_consumed", 0.0) or 0.0)
                self.quality = {
                    "energy": "measured",
                    "carbon": self._carbon_quality_for_measured_energy(),
                }
                self._tool = "codecarbon"
            except Exception:
                self._tracker = None

        # CodeCarbon can't read power on Apple Silicon / many laptops and returns
        # ~0. Don't launder a zero as "measured" -- fall back to the NVML reading.
        if (self.energy_kwh or 0.0) <= 0.0 and nvml_kwh > 0.0:
            self.energy_kwh = nvml_kwh * self.pue
            self.carbon_kg = self.energy_kwh * self.ci
            self.quality = {
                "energy": "measured",
                "carbon": self._carbon_quality_for_measured_energy(),
            }
            self._tool = "nvml"

        if (self.energy_kwh or 0.0) <= 0.0:  # TDP estimate
            tdp = _tdp_for(self.gpu)
            # CPU per-core figure already reflects under-load draw; the utilization
            # factor only applies to a GPU's rated TDP.
            util = _gpu_utilization_factor()
            pavg = tdp if self.is_cpu else tdp * util
            self.energy_kwh = gpu_hours * pavg * self.pue / 1000.0
            self.carbon_kg = self.energy_kwh * self.ci
            self.quality = {"energy": "estimated-from-hardware", "carbon": "estimated-from-region"}
            self._tool = "dia-track-estimate"

        self.gpu_hours = gpu_hours
        energy = self.energy_kwh or 0.0
        self.water_l = (energy * self.wue[0], energy * self.wue[1])
        self.quality["water"] = "estimated-from-region" if self._wue_user_supplied else "estimated-from-default-wue"
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
                "tool": self._tool,
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
