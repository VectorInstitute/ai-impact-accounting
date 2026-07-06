# Sustainable Open-Source AI Requires Tracking the Cumulative Footprint of Derivatives

**ICML 2026 Position Paper Track — Spotlight** (top 5% of submissions)

📄 [Paper (arXiv)](https://arxiv.org/abs/2601.21632) · 🌐 [Project Page](https://vectorinstitute.github.io/ai-impact-accounting/)

**Authors:** Shaina Raza, Iuliia Zarubiieva, Ahmed Y. Radwan, Nate Lesperance, Deval Pandya, Sedef Akinli Kocak, Graham W. Taylor

*Vector Institute for Artificial Intelligence / University of Guelph*

---

## Overview

Open-source AI has a hidden cumulative footprint problem. Hugging Face now hosts over 2 million models, and a single foundation model like Llama can spawn hundreds of fine-tunes, LoRA adapters, quantizations, and forks within months of release. Each derivative consumes energy and water that goes largely untracked.

Compute efficiency alone is not enough. Lower per-run costs encourage more experimentation and deployment, which can increase aggregate emissions even as individual runs get cheaper. This is the rebound effect, and it means the open-source ecosystem is collectively exposed to a tragedy of the commons: individually rational actions accumulate into a shared environmental cost that no single actor observes or manages.

## Key Numbers

- Global data centre electricity is projected to grow from **415 TWh (2024) to 945 TWh by 2030**, a 15% annual rate
- AI-specific servers are growing at **30% annually**
- Pretraining Llama 3 emitted approximately **2,290 tCO2eq**; one documented model family already has **146+ derivatives**
- GPT-4 training is estimated at **11,000–18,870 tCO2eq** and **76–170 megalitres** of water consumed
- One in four data centre assets may face increased water scarcity by 2050

## Proposal: Data and Impact Accounting (DIA)

DIA is a lightweight, voluntary transparency layer with three components:

1. **Standardized impact schema** embedded in model cards: hardware, GPU-hours, estimated kWh, CO2, water use, and model lineage
2. **Low-friction instrumentation** via existing tools (CodeCarbon, ML CO2 Impact, cloud provider APIs) integrated into training pipelines
3. **Ecosystem dashboards** that aggregate reported footprints across model families and derivatives over time

DIA is non-regulatory. It does not restrict who can train or release models. The goal is visibility into trends and relative impacts, not auditing individual projects.

## Using DIA from PyPI

Install the library (Python 3.12+):

```bash
pip install ai-impact-accounting

# optional extras
pip install "ai-impact-accounting[measure]"    # CodeCarbon / NVML when available
pip install "ai-impact-accounting[dashboard]"  # local Gradio viewer
```

The wheel ships `ai_impact_accounting` and the `dia` CLI. Runnable training demos
live under `scripts/` in this repo — clone the repo to run those, or wire DIA into
your own trainer as below.

### Instrument a training run

```python
from ai_impact_accounting import track

with track(base_model="distilbert-base-uncased", relation="finetune") as t:
    trainer.train()

t.write("out/README.md")              # local model card + dia_report
t.push("your-org/your-model")         # update the Hub card (needs HF token)
```

Set grid context before training (optional but recommended):

```bash
export DIA_REGION=ca-on
export DIA_CI=0.03
```

### Ctrl+C and partial runs (PyPI-only integrators)

`track()` samples energy for the whole `with` block and finalizes metrics on exit.
It does **not** save your weights or write a card unless you call `t.write()` /
`t.push()` yourself. On **Ctrl+C**, use the same pattern as `scripts/dia_finalize.py`:

```python
interrupted = False
with track(base_model="...", relation="finetune") as t:
    try:
        trainer.train()
    except KeyboardInterrupt:
        interrupted = True

trainer.save_model("out/")
t.write("out/README.md")
if not interrupted:
    t.push("your-org/your-model")
```

For local-only runs (no Hub push): `export DIA_LOCAL=1` is honoured if you copy
the helpers from `scripts/dia_finalize.py`, or simply skip `t.push()`.

### CLI

```bash
dia validate path/to/README.md    # or a Hub repo id
dia report   path/to/README.md
```

### Repo demos vs the installed package

| | PyPI install | This repo (`scripts/`) |
|---|--------------|-------------------------|
| `track()`, `dia`, `Store`, ingest | yes | yes (editable install) |
| Training demos + `dia_finalize.py` | no — clone repo | yes |
| Full lab (A100/A40/CPU, ingest, dashboard) | see **[LAB.md](LAB.md)** | **[LAB.md](LAB.md)** |

### Lab workflow

See **[LAB.md](LAB.md)** for setup, A100 / A40 / CPU training, ingest, and the Gradio dashboard.

## Project Page

The `docs/` folder contains the companion website (`index.html`, `style.css`, `main.js`) with an interactive model footprint table, carbon vs. water scatter plot, and DIA overview. It is a static site with no build step required and is published as the [project page](https://vectorinstitute.github.io/ai-impact-accounting/).

## Acknowledgements

Resources used in preparing this research were provided in part by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute (vectorinstitute.ai/#partners).

Graham W. Taylor acknowledges support from the Natural Sciences and Engineering Research Council (NSERC), the Canada Research Chairs program, and the Canadian Institute for Advanced Research (CIFAR) Canada CIFAR AI Chairs program.

## Citation

```bibtex
@article{raza2026sustainable,
  title={Sustainable Open-Source AI Requires Tracking the Cumulative Footprint of Derivatives},
  author={Raza, Shaina and Zarubiieva, Iuliia and Radwan, Ahmed Y and Lesperance, Nate and Pandya, Deval and Kocak, Sedef Akinli and Taylor, Graham W},
  journal={arXiv preprint arXiv:2601.21632},
  year={2026}
}
```
