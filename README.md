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

### Using your own Hugging Face namespace

The runnable demos under `scripts/` default to the Vector Institute's demo org
(**`DIA-MVP`**) so the paper demo works out of the box. The **library itself
has no `DIA-MVP` dependency** — pass your own repo ids in API calls or override
the demo defaults with environment variables.

| | Demo defaults (`scripts/`) | Your setup |
|---|---|---|
| Model repo | `DIA-MVP/my-bert-sentiment` | `your-username/your-model` via `t.push(...)` or `REPO=…` |
| State dataset | `DIA-MVP/dia-state` | `your-username/dia-state` via `Store(...)` or `DIA_DATASET=…` |
| Base model tracked | `distilbert-base-uncased` | Your base via `DIA_BASES=…` |

**Report a footprint** — push to any model repo you own:

```python
from ai_impact_accounting import track

with track(base_model="meta-llama/Llama-3-8B", relation="qlora") as t:
    train(...)

t.push("your-username/my-llama-finetune")
```

**Aggregate and visualize** — use your own state dataset (created automatically
on first use). A bare name expands to `<your-hf-username>/dia-state`:

```python
from ai_impact_accounting import Store, ingest_model, rollup

store = Store("dia-state")                              # → your-username/dia-state
ingest_model("your-username/my-llama-finetune", store)
res = rollup(store.nodes, "meta-llama/Llama-3-8B")
```

**Run the BERT demo against your repos** (after `hf auth login`):

```bash
REPO=your-username/my-bert-sentiment python scripts/train_bert_demo.py

export DIA_DATASET=your-username/dia-state
export DIA_BASES=distilbert-base-uncased
./scripts/run_local.sh your-username/my-bert-sentiment
```

**Deploy a hosted Space** — set secrets to your repos, not `DIA-MVP`:

| Secret | Example | Purpose |
|---|---|---|
| `DIA_DATASET` | `your-username/dia-state` | Where ingested footprints are stored |
| `DIA_BASES` | `meta-llama/Llama-3-8B` | Comma-separated base models to track |
| `HF_TOKEN` | your write token | Read model cards, write state |
| `WEBHOOK_SECRET` | random string | Auto-ingest on model push |

### Install

```bash
pip install ai-impact-accounting
```

The base install is light and gives you the producer instrumentation, the `dia`
CLI, and the aggregator. Heavier, optional capabilities are extras:

| Extra | `pip install "ai-impact-accounting[...]"` | What it adds |
|---|---|---|
| `measure` | `[measure]` | CodeCarbon for **measured** (not estimated) energy/carbon during training |
| `dashboard` | `[dashboard]` | Gradio dashboard + Hugging Face webhook Space (self-hosting) |
| `examples` | `[examples]` | torch / transformers / peft / datasets for the demos in `scripts/` |
| `all` | `[all]` | `measure` + `dashboard` |

Extras combine, e.g. `pip install "ai-impact-accounting[measure,dashboard]"`.

### Local development

To hack on this repo or smoke-test the BERT demo against your checkout (no PyPI
publish needed), clone the repo, create a venv, and install **from the local
tree** with [uv](https://docs.astral.sh/uv/):

```bash
cd ai-impact-accounting
uv venv && source .venv/bin/activate
uv sync --extra examples --extra measure --extra dashboard --dev
```

This installs the package in editable mode from `src/` plus the demo extras
(`torch`, `transformers`, `gradio`, …) and dev tools (`pytest`, `ruff`, …).
Re-run `uv sync …` after dependency or refactor changes.

Equivalent with pip (still local, not PyPI):

```bash
pip install -e ".[examples,measure,dashboard]"
```

Sanity check that Python is importing your checkout:

```bash
python -c "import ai_impact_accounting; print(ai_impact_accounting.__file__)"
# should print …/ai-impact-accounting/src/ai_impact_accounting/__init__.py
```

End-to-end BERT demo (requires `hf auth login` once). By default this
pushes to the shared demo org `DIA-MVP`; see **Using your own Hugging Face
namespace** above to use your account instead.

```bash
python scripts/train_bert_demo.py   # train + push a model with a dia_report
./scripts/run_local.sh              # ingest it + launch the Gradio dashboard
```

On Apple Silicon, skip the `measure` extra unless you want CodeCarbon (it prompts
for `sudo` via `powermetrics` and usually falls back to a TDP estimate anyway):

```bash
uv sync --extra examples --extra dashboard --dev
```

Faster checks without a full training run:

```bash
pytest tests/ -m "not integration_test"
dia validate distilbert-base-uncased
```

If you already pushed `DIA-MVP/my-bert-sentiment` in a prior run, skip training
and go straight to `./scripts/run_local.sh`.

### Report a footprint (producer)

Wrap your training loop. The only DIA-specific lines are `track(...)` and
`push(...)`; everything else is your normal pipeline. `track()` auto-detects
hardware, measures with CodeCarbon when available (otherwise falls back to a
hardware-TDP estimate), and stamps every field with its data-quality tier.

```python
from ai_impact_accounting import track

with track(base_model="meta-llama/Llama-3-8B", relation="qlora") as t:
    train(...)

t.push("you/llama3-8b-mydataset")   # merges a dia_report block into the model card
# or, before push_to_hub on a local card:  t.write("README.md")
```

### Validate a card (CLI)

A drop-in check for CI or conference submissions:

```bash
dia validate you/llama3-8b-mydataset   # OK / FAIL against the DIA schema
dia report   you/llama3-8b-mydataset   # print the parsed footprint
```

`dia validate` also accepts a local card path (e.g. `dia validate README.md`).

### Aggregate a family (core)

Given collected nodes, roll up a base model's whole subtree. Lineage is treated
as a DAG, so merges and shared ancestors are summed exactly once; the result
reports a lower bound plus coverage and per-field provenance, never a bare total.

```python
from ai_impact_accounting import Store, ingest_model, rollup

store = Store("you/dia-state")                 # state lives in a HF dataset repo
ingest_model("you/llama3-8b-mydataset", store)
res = rollup(store.nodes, "meta-llama/Llama-3-8B")
print(res["coverage"], res["total_footprint"]["carbon"].fmt(" kgCO2eq"))
```

### Self-host the dashboard (operator)

Requires the `dashboard` extra. Locally (see **Local development** for `uv sync`
setup and **Using your own Hugging Face namespace** for repo configuration):

```bash
python scripts/train_bert_demo.py     # train + push a model carrying a dia_report
./scripts/run_local.sh                # ingest it + launch the Gradio dashboard
```

To deploy the live Space (webhook + nightly crawler + dashboard), push this repo
to a Docker Space and run `ai_impact_accounting.dashboard.app`. Set Space
secrets to **your** dataset and base models (see table in **Using your own
Hugging Face namespace**). The webhook route is mounted under `/webhooks`, so
the registered URL is doubled:
`https://<space>.hf.space/webhooks/webhooks/ingest`.

### Repo layout

```
src/ai_impact_accounting/
├── models.py · parse.py · graph.py   # core: interval math, parsing, DAG rollup
├── schema/dia_schema.json            # the dia_report JSON Schema (packaged)
├── producer/  tracking.py · cli.py   # track() + the `dia` CLI
├── hub/       store.py · ingest.py · crawl.py   # collect reports (extra-light)
└── dashboard/ ui.py · app.py         # Gradio + HF webhook Space  [extra: dashboard]
scripts/                              # runnable demos + local dashboard launcher
tests/                                # unit tests; Hub-touching tests marked integration_test
```

The `scripts/` demos each exercise a different training regime, so you can see
`track()` across frameworks, lineage shapes, and footprint scales:

| Script | Regime | Shows |
|---|---|---|
| `train_bert_demo.py` | DistilBERT SST-2 fine-tune (HF `Trainer`) | a `finetune` derivative |
| `train_llama_lora.py` | TinyLlama / Llama LoRA (PEFT) | a small `lora` derivative |
| `train_resnet50_cifar.py` | Full fine-tune of pretrained ResNet-50 on CIFAR-100 (time-budgeted) | a high-energy `finetune` derivative; ~30-60 min on an M-series Mac |
| `train_simclr_cifar.py` | SimCLR self-supervised pretrain on CIFAR-10 | `track()` around a raw PyTorch loop; a from-scratch family root |
| `view_local.py` / `run_local.sh` | — | ingest a model + launch the dashboard locally |

### Correctness rules (the parts that are easy to get wrong)

- **`scope: incremental`** — each node reports only its own delta; the family total is the subtree sum. Cumulative scope is rejected at parse time so shared ancestors aren't double-counted.
- **Lineage is a DAG, not a tree** — a merge has multiple parents; rollup sums each unique node once (visited-set traversal), deduping merges and shared ancestors automatically.
- **Coverage is the headline** — disclosure starts at ~10–40%, so totals are a lower bound + coverage %; imputation of missing nodes is opt-in and always labelled `imputed`.
- **Per-field quality tiers** — `measured` vs `estimated-*` vs `imputed`, kept segregated so the aggregator never sums apples and oranges.

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
