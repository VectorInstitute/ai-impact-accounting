#!/usr/bin/env python3
"""Fill in DIA lab model cards and the dataset card on Hugging Face.

Every lab model repo carries a ``dia_report`` block but an empty/auto-generated
body. This regenerates a useful description for each (preserving ``dia_report``
and enriching front-matter), and writes a dataset card for the rollup table.

Usage:
    python scripts/update_cards.py            # dry-run: print generated cards
    python scripts/update_cards.py --push     # write to Hugging Face
"""

from __future__ import annotations

import argparse
import re
import sys

import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

ORG = "DIA-MVP"
DATASET = f"{ORG}/dia-state-lab-2026"
DASHBOARD_NOTE = (
    "This footprint feeds the DIA dashboard, which rolls up a base model and all "
    "its derivatives to show the **cumulative** carbon, water, and energy cost of a "
    "model family."
)

# Per-model spec, keyed by repo name with the hardware suffix stripped.
SPECS: dict[str, dict] = {
    "my-bert-sentiment": {
        "title": "DistilBERT — SST-2 sentiment",
        "task": "binary sentiment classification (SST-2)",
        "base": "distilbert-base-uncased",
        "method": "full fine-tune",
        "pipeline_tag": "text-classification",
        "license": "apache-2.0",
        "library": "transformers",
        "script": "scripts/train_bert_demo.py",
    },
    "llama32-3b-lora": {
        "title": "Llama 3.2 3B Instruct — LoRA",
        "task": "instruction-tuning (LoRA adapter)",
        "base": "meta-llama/Llama-3.2-3B-Instruct",
        "method": "LoRA (PEFT)",
        "pipeline_tag": "text-generation",
        "license": "llama3.2",
        "library": "peft",
        "script": "scripts/train_llama_lora.py",
    },
    "qwen2.5-7b-lora": {
        "title": "Qwen2.5-7B — LoRA",
        "task": "instruction-tuning (LoRA adapter)",
        "base": "Qwen/Qwen2.5-7B",
        "method": "LoRA (PEFT)",
        "pipeline_tag": "text-generation",
        "license": "apache-2.0",
        "library": "peft",
        "script": "scripts/train_qwen_lora.py",
    },
    "tinyllama-lora": {
        "title": "TinyLlama 1.1B Chat — LoRA",
        "task": "instruction-tuning (LoRA adapter)",
        "base": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "method": "LoRA (PEFT)",
        "pipeline_tag": "text-generation",
        "license": "apache-2.0",
        "library": "peft",
        "script": "scripts/train_llama_lora.py",
    },
    "resnet50-cifar100": {
        "title": "ResNet-50 — CIFAR-100",
        "task": "image classification (CIFAR-100)",
        "base": "microsoft/resnet-50",
        "method": "full fine-tune",
        "pipeline_tag": "image-classification",
        "license": "apache-2.0",
        "library": None,
        "script": "scripts/train_resnet50_cifar.py",
    },
    "cifar10-simclr": {
        "title": "SimCLR ResNet-18 — CIFAR-10",
        "task": "self-supervised contrastive pretraining (CIFAR-10)",
        "base": "scratch",
        "method": "from-scratch SimCLR encoder",
        "pipeline_tag": None,
        "license": "apache-2.0",
        "library": None,
        "script": "scripts/train_simclr_cifar.py",
    },
    "mnist-ddpm": {
        "title": "DDPM — MNIST",
        "task": "denoising diffusion generative model (MNIST)",
        "base": "scratch",
        "method": "from-scratch DDPM",
        "pipeline_tag": None,
        "license": "apache-2.0",
        "library": None,
        "script": "scripts/train_ddpm_mnist.py",
    },
}

HW_LABEL = {"a100": "NVIDIA A100", "a40": "NVIDIA A40", "cpu": "CPU (80-core)"}

REPOS = [
    "my-bert-sentiment-a100", "llama32-3b-lora-a100", "qwen2.5-7b-lora-a100",
    "resnet50-cifar100-a100", "cifar10-simclr-a100", "mnist-ddpm-a100", "tinyllama-lora-a100",
    "my-bert-sentiment-a40", "tinyllama-lora-a40", "cifar10-simclr-a40",
    "my-bert-sentiment-cpu", "tinyllama-lora-cpu", "mnist-ddpm-cpu",
]


def _spec_for(repo_name: str) -> tuple[dict, str]:
    """Return (spec, hardware) for a repo name like ``my-bert-sentiment-a100``."""
    for hw in ("a100", "a40", "cpu"):
        if repo_name.endswith(f"-{hw}"):
            return SPECS[repo_name[: -(len(hw) + 1)]], hw
    raise KeyError(repo_name)


def _split_front(text: str) -> tuple[dict, str]:
    """Split YAML front-matter (first block only) from the body."""
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.DOTALL)
    if m:
        return (yaml.safe_load(m.group(1)) or {}), m.group(2)
    return {}, text


def _fp_table(dia: dict) -> str:
    """Render the footprint as a small markdown table from a dia_report dict."""
    fp = dia.get("footprint", {})
    comp = dia.get("compute", {})
    hw = comp.get("hardware", {})

    def val(d: dict) -> str:
        v = d.get("value")
        q = d.get("quality", "")
        if isinstance(v, list):
            s = f"{v[0]}–{v[1]}"
        else:
            s = f"{v}"
        return f"{s} ({q})" if q else s

    gpu = hw.get("gpu", "?")
    n = hw.get("count", 1)
    hrs = comp.get("duration_gpu_hours", "?")
    region = dia.get("context", {}).get("region", "?")
    return (
        f"| Metric | Value |\n|---|---|\n"
        f"| Hardware | {n}× {gpu} |\n"
        f"| Compute | {hrs} GPU-hours |\n"
        f"| Energy | {val(fp.get('energy_kwh', {}))} kWh |\n"
        f"| Carbon | {val(fp.get('carbon_kgco2eq', {}))} kgCO₂eq |\n"
        f"| Water | {val(fp.get('water_liters', {}))} L |\n"
        f"| Grid region | {region} |\n"
    )


def build_card(repo_name: str, raw: str) -> str:
    """Build the full model card text (front-matter + body) for one repo."""
    spec, hw = _spec_for(repo_name)
    front, _ = _split_front(raw)
    dia = front.get("dia_report", {})

    # Enrich front-matter, preserving dia_report verbatim.
    front.setdefault("tags", [])
    for t in ("dia", "carbon-footprint", "energy-efficiency", "sustainability"):
        if t not in front["tags"]:
            front["tags"].append(t)
    if spec.get("license"):
        front["license"] = spec["license"]
    if spec.get("library"):
        front["library_name"] = spec["library"]
    if spec.get("pipeline_tag"):
        front["pipeline_tag"] = spec["pipeline_tag"]
    if spec["base"] != "scratch":
        front["base_model"] = spec["base"]

    base_txt = "trained from scratch" if spec["base"] == "scratch" else f"`{spec['base']}`"
    repo_id = f"{ORG}/{repo_name}"
    body = f"""# {spec['title']} ({HW_LABEL[hw]})

A demo model from the **Data & Impact Accounting (DIA)** lab. It performs
{spec['task']} via **{spec['method']}**, with the base model {base_txt}, trained on
**{HW_LABEL[hw]}**.

The point of this repo is not the model itself but its **`dia_report`** — a
standardized record of the energy, carbon, and water used to train it, embedded
in this card's metadata.

{DASHBOARD_NOTE}

## Training footprint

{_fp_table(dia)}
*Energy and carbon are measured with [CodeCarbon](https://github.com/mlco2/codecarbon);
water is estimated from a default water-usage-effectiveness range. Carbon uses the
local grid's intensity (Ontario, ~0.03 kgCO₂eq/kWh).*

## Reproduce

```bash
REPO={repo_id} python {spec['script']}
```

## Links

- **Footprint table (dataset):** [{DATASET}](https://huggingface.co/datasets/{DATASET})
- **Project / paper:** [ai-impact-accounting](https://github.com/VectorInstitute/ai-impact-accounting)
- **Lab workflow:** see `LAB.md` in the repo
"""
    fm = yaml.safe_dump(front, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return f"---\n{fm}---\n\n{body}"


def build_dataset_card() -> str:
    """Build the dataset card for the rollup table."""
    front = {
        "license": "apache-2.0",
        "tags": ["dia", "carbon-footprint", "energy-efficiency", "sustainability", "model-lineage"],
        "pretty_name": "DIA lab footprint table (2026)",
    }
    body = f"""# DIA lab footprint table

This dataset is the **rollup table** for the Data & Impact Accounting (DIA) lab
demo. It indexes the training footprint (energy, carbon, water) and lineage of a
set of demo models trained across **A100 / A40 / CPU** hardware.

It is produced by ingesting each model's `dia_report` card and is read by the DIA
web dashboard. It stores **metadata only — no model weights**.

## Files

- **`nodes.parquet`** — one flat row per model (browsable in the Dataset Viewer):
  energy/carbon/water intervals, data-quality tier, GPU, GPU-hours, region, lineage.
- **`state.json`** — the nested source of truth the dashboard loads.

## How the rollup works

Given a **base model**, the dashboard builds the lineage as a directed graph and
takes the base plus all its descendants as the *family*, then:

1. **Sums incremental footprints** — each model logs only its own training delta;
   the family total is the subtree sum.
2. **Dedupes the DAG** — a merged/shared model is counted once.
3. **Reports coverage, not a bare total** — totals are a lower bound at low disclosure.
4. **Keeps provenance separate** — `measured` vs `estimated` vs `imputed`.

## Related

- **Toolkit / paper:** [VectorInstitute/ai-impact-accounting](https://github.com/VectorInstitute/ai-impact-accounting)
- **Models:** the `{ORG}/*-a100`, `*-a40`, `*-cpu` repos ingested here.
"""
    fm = yaml.safe_dump(front, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return f"---\n{fm}---\n\n{body}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Fill DIA model + dataset cards.")
    ap.add_argument("--push", action="store_true", help="Write to Hugging Face (default: dry-run).")
    ap.add_argument("--only", help="Only process this repo name (model) for preview.")
    args = ap.parse_args()
    api = HfApi()

    repos = [args.only] if args.only else REPOS
    for repo_name in repos:
        repo_id = f"{ORG}/{repo_name}"
        try:
            raw = open(hf_hub_download(repo_id, "README.md", repo_type="model")).read()
        except (EntryNotFoundError, RepositoryNotFoundError) as e:
            print(f"SKIP {repo_id}: {e}")
            continue
        card = build_card(repo_name, raw)
        if args.push:
            api.upload_file(
                path_or_fileobj=card.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add model description (DIA lab card)",
            )
            print(f"PUSHED {repo_id}")
        else:
            print(f"\n{'=' * 70}\n{repo_id}\n{'=' * 70}\n{card}")

    # Dataset card
    ds_card = build_dataset_card()
    if args.push:
        api.upload_file(
            path_or_fileobj=ds_card.encode(),
            path_in_repo="README.md",
            repo_id=DATASET,
            repo_type="dataset",
            commit_message="Add dataset card (DIA rollup table)",
        )
        print(f"PUSHED dataset {DATASET}")
    elif not args.only:
        print(f"\n{'=' * 70}\nDATASET {DATASET}\n{'=' * 70}\n{ds_card}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
