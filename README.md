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

## Getting started with DIA

DIA ships as the `ai_impact_accounting` Python package and `dia` CLI. **Until the
package is on PyPI, clone this repo** for training demos (`scripts/`), ingest,
crawl, and the web dashboard. A minimal `track()` integration works from an
editable install; the full lab workflow is in **[LAB.md](LAB.md)**.

### Install (from a clone)

Requires **Python 3.12+**.

```bash
git clone https://github.com/VectorInstitute/ai-impact-accounting.git
cd ai-impact-accounting

# recommended — reproducible lockfile
uv sync --all-extras --dev
source .venv/bin/activate

# alternative: pip editable install
# python -m venv .venv && source .venv/bin/activate
# pip install -e ".[measure,dashboard,examples]"

hf auth login   # once; needs a write token for Hub push / ingest
```

| Extra | Install flag | What it adds |
|-------|----------------|--------------|
| Core | (default) | `track()`, `dia`, lineage DAG, Hub ingest |
| Measured runs | `[measure]` | CodeCarbon / NVML when available |
| Dashboard | `[dashboard]` | Local FastAPI web UI (`scripts/view_local.py`) |
| Training demos | `[examples]` | Dependencies for `scripts/train_*.py` |

**PyPI:** `pip install ai-impact-accounting` will be supported once the release is
published; until then use the clone + editable install above.

### Instrument a training run

```python
from ai_impact_accounting import track

with track(base_model="distilbert-base-uncased", relation="finetune") as t:
    trainer.train()

t.write("out/README.md")              # local model card + dia_report
t.push("your-org/your-model")         # update the Hub card (needs HF token)
```

```bash
export DIA_REGION=ca-on
export DIA_CI=0.03
```

### Run training demos (this repo)

After install + `hf auth login`:

```bash
export DIA_REGION=ca-on
export DIA_CI=0.03
export DIA_DATASET=DIA-MVP/dia-state-lab-2026

REPO=your-org/my-bert-sentiment python scripts/train_bert_demo.py
```

All demos share `scripts/dia_finalize.py` for Ctrl+C / partial-run handling.
See **[LAB.md](LAB.md)** for A100 / A40 / CPU recipes, ingest, and batch jobs.

<details>
<summary><strong>Environment variables</strong></summary>

Set these in every shell (or add to your job script). Training scripts and the
dashboard read them from the environment — there is no separate config file.

**Grid / footprint context (training)**

| Variable | Required | Example | Purpose |
|----------|----------|---------|---------|
| `DIA_REGION` | recommended | `ca-on` | Grid region label on the `dia_report` |
| `DIA_CI` | optional | `0.03` | **C**arbon **i**ntensity of the grid (kgCO₂/kWh); used with energy to estimate emissions; default `0.40` if omitted |
| `HF_TOKEN` | for Hub push | — | Write token; or use `hf auth login` |

`DIA_CI` is not measured during the run — you set it to match where the GPUs ran
(e.g. `0.03` for a low-carbon grid like Ontario, `0.45` for a higher-carbon US
grid). GPU-hours and kWh are comparable across sites; carbon and water scale with
your grid. See [LAB.md](LAB.md) for region-contrast examples.

**Training run control (`scripts/`)**

| Variable | Required | Example | Purpose |
|----------|----------|---------|---------|
| `REPO` | yes | `your-org/my-bert-sentiment` | Target Hub model repo id |
| `BASE` | some scripts | `distilbert-base-uncased` | Weights to load (LoRA / finetune parent) |
| `LINEAGE_MODEL` | optional | `your-org/parent-adapter` | Parent recorded in `dia_report.lineage` (defaults to `BASE`) |
| `DIA_LOCAL=1` | optional | `1` | Save under `out-*/`, write card locally, skip Hub push |
| `DIA_NO_PUSH=1` | optional | `1` | Same as local-only for this run; still uses `REPO` as dashboard hint |

**Dataset, dashboard, and crawl**

| Variable | Required | Example | Purpose |
|----------|----------|---------|---------|
| `DIA_DATASET` | for dashboard / ingest | `DIA-MVP/dia-state-lab-2026` | HF dataset repo holding `state.json` |
| `DIA_BASES` | optional | `distilbert-base-uncased` | Default base model in the UI (comma-separated) |
| `DIA_STATE_FILE` | optional | `tests/fixtures/synth-100-state.json` | Local `state.json` instead of Hub (stress tests) |
| `DIA_ORGS` | crawl | `vector-institute,DIA-MVP` | Orgs/users to index (nightly crawl) |
| `DIA_SPACE` | optional | `vector-institute/dia-dashboard` | Restart Space after crawl |
| `PORT` / `HOST` | optional | `7860` / `0.0.0.0` | Local dashboard bind address |
| `WEBHOOK_SECRET` | Space webhook | — | Shared secret for ingest webhook (see [LAB.md](LAB.md)) |

Public lab defaults: dataset **[DIA-MVP/dia-state-lab-2026](https://huggingface.co/datasets/DIA-MVP/dia-state-lab-2026)**,
dashboard **[DIA-MVP/dia-dashboard](https://huggingface.co/spaces/DIA-MVP/dia-dashboard)**.
Use your own `DIA_DATASET` / namespace if you do not have write access to `DIA-MVP/*`.

</details>

<details>
<summary><strong>Local web dashboard &amp; Space deploy</strong></summary>

```bash
export DIA_DATASET=DIA-MVP/dia-state-lab-2026
export DIA_BASES=distilbert-base-uncased
python scripts/view_local.py
```

Open **http://127.0.0.1:7860**. `HF_TOKEN` is optional for read-only public datasets.

Or ingest one model then launch:

```bash
./scripts/run_local.sh your-org/my-bert-sentiment
```

Deploy or update the Hugging Face Space:

```bash
python scripts/deploy_space.py
# or: python scripts/deploy_space.py --space your-org/dia-dashboard --dataset your-org/dia-state
```

</details>

<details>
<summary><strong>Ctrl+C and partial runs</strong></summary>

`track()` samples energy for the whole `with` block and finalizes metrics on exit.
It does **not** save your weights or write a card unless you call `t.write()` /
`t.push()` yourself. Repo demos use `scripts/dia_finalize.py` on **Ctrl+C**; for
your own trainer, copy that pattern or set `DIA_LOCAL=1` and skip `t.push()`.

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

</details>

<details>
<summary><strong>CLI</strong></summary>

```bash
dia validate path/to/README.md    # or a Hub repo id
dia report   path/to/README.md
```

</details>

<details>
<summary><strong>What lives where (repo vs PyPI)</strong></summary>

| | Editable install (this repo) | PyPI (coming soon) |
|---|------------------------------|---------------------|
| `track()`, `dia`, `Store`, ingest | yes | yes |
| `scripts/` training demos + `dia_finalize.py` | yes | no — clone repo |
| Web dashboard, crawl, Space deploy | yes | `[dashboard]` extra only |
| Full lab (GPU tiers, ingest, HF Space) | **[LAB.md](LAB.md)** | **[LAB.md](LAB.md)** |

</details>

### Lab workflow

See **[LAB.md](LAB.md)** for setup, A100 / A40 / CPU training, ingest, nightly
crawl, webhooks, and the web dashboard (local or [HF Space](https://huggingface.co/spaces/DIA-MVP/dia-dashboard)).

## Project Page

The `docs/` folder contains the companion website (`index.html`, `style.css`, `main.js`) with an interactive model footprint table, carbon vs. water scatter plot, DIA overview, and an embedded read-only dashboard (`?embed=1`). It is a static site with no build step required and is published as the [project page](https://vectorinstitute.github.io/ai-impact-accounting/).

## Acknowledgements

This research was funded by the European Union's Horizon Europe research and innovation programme under the AIXPERT project (Grant Agreement No. 101214389), which aims to develop an agentic, multi-layered, GenAI-powered framework for creating explainable, accountable, and transparent AI systems.

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
