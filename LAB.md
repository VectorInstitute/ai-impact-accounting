# DIA lab workflow (DIA-MVP)

Train demo models on **A100 / A40 / CPU**, push `dia_report` to Hugging Face model
repos, ingest into **[DIA-MVP/dia-state-lab-2026](https://huggingface.co/datasets/DIA-MVP/dia-state-lab-2026)**,
and view in the **web dashboard** (local or [HF Space](https://huggingface.co/spaces/DIA-MVP/dia-dashboard)).

Public crawl data stays in [`DIA-MVP/dia-state`](https://huggingface.co/datasets/DIA-MVP/dia-state) — do not mix the two.

---

## Model repo vs DIA table

| | Hugging Face | Holds |
|---|--------------|--------|
| **Model repo** | `DIA-MVP/my-bert-sentiment-a100` | Weights + `dia_report` on the card |
| **DIA table** | `DIA-MVP/dia-state-lab-2026` | `state.json` + `nodes.parquet` (metadata only) |

Training → model repo. **Ingest** → table. Weights are never stored in the table.

---

## Setup (every shell)

First-time install: see **[README.md — Install](README.md#getting-started-with-dia)** (`git clone`, `uv sync --all-extras --dev`, `hf auth login`).

```bash
cd /path/to/ai-impact-accounting
source .venv/bin/activate
hf auth login   # once

export DIA_REGION=ca-on
export DIA_CI=0.03
export DIA_DATASET=DIA-MVP/dia-state-lab-2026
```

GPU type is **auto-detected** in `dia_report`. Set it by **which node you use** and
by **`REPO=` suffix** (`-a100`, `-a40`, `-cpu`).

**Other locations:** `DIA_REGION` and `DIA_CI` are **your grid**, not Vector’s cluster.
Use a label you recognize (e.g. `eu-de`, `us-east-1`, `au-nsw`) and your grid’s carbon
intensity in kgCO₂/kWh. If you omit `DIA_CI`, DIA uses **0.40** (generic default).
Energy and GPU-hours are comparable across sites; **carbon and water** scale with your
grid. Use **your own HF namespace** and `REPO=` names — do not push into `DIA-MVP/*`
unless you have write access. Slurm/`srun` flags depend on your HPC, not ours.

### Interrupts and local-only runs

All training scripts under `scripts/` share `dia_finalize.py`. If you press
**Ctrl+C** (or the job hits a time limit), the script still saves local weights,
writes a `dia_report` into `out-*/README.md`, and prints `dia validate …` so
you can check the card. Partial runs exit with code **130**.

**Local-only (no Hub push):**

```bash
DIA_LOCAL=1 REPO=my-local-run python scripts/train_bert_demo.py
```

**Push later, but skip push for this run:**

```bash
DIA_NO_PUSH=1 REPO=DIA-MVP/my-bert-sentiment-cpu python scripts/train_bert_demo.py
```

| Variable | Effect |
|----------|--------|
| `DIA_LOCAL=1` | Save under `out-*/`, write `dia_report`, skip Hub push |
| `DIA_NO_PUSH=1` | Same skip-push behaviour; `REPO` is still used as the dashboard hint |
| (no token) | Local save only; push is skipped with a clear message |
| Hub push error | Local artifacts kept; a warning is printed |

After a local run, validate and inspect the report:

```bash
dia validate out-bert/README.md
dia report   out-bert/README.md
```

To show the run in the dashboard, ingest the local card (or push to a model repo
first, then ingest that repo id). See **Ingest** and **Web dashboard** below.

**PyPI-only installs:** the wheel does not include `scripts/` or `dia_finalize.py`.
See **[README.md — Getting started with DIA](README.md#getting-started-with-dia)** for
clone + editable install, environment variables, and the interrupt/finalize pattern.

---

## A100 (7 models)

```bash
srun --nodelist=bn099 --gres=gpu:a100:1 --mem=32G --time=8:00:00 --pty bash
```

Run **one at a time**:

```bash
REPO=DIA-MVP/my-bert-sentiment-a100 python scripts/train_bert_demo.py

BASE=meta-llama/Llama-3.2-3B-Instruct \
REPO=DIA-MVP/llama32-3b-lora-a100 N_EXAMPLES=5000 EPOCHS=1 \
python scripts/train_llama_lora.py

MAX_STEPS=1000 REPO=DIA-MVP/qwen2.5-7b-lora-a100 python scripts/train_qwen_lora.py

REPO=DIA-MVP/resnet50-cifar100-a100 MAX_MINUTES=40 python scripts/train_resnet50_cifar.py

REPO=DIA-MVP/cifar10-simclr-a100 EPOCHS=10 N_TRAIN=15000 python scripts/train_simclr_cifar.py

REPO=DIA-MVP/mnist-ddpm-a100 MAX_MINUTES=30 python scripts/train_ddpm_mnist.py

REPO=DIA-MVP/tinyllama-lora-a100 python scripts/train_llama_lora.py
```

---

## A40 (3 models)

```bash
srun --gres=gpu:a40:1 --mem=32G --time=4:00:00 --pty bash

REPO=DIA-MVP/my-bert-sentiment-a40 python scripts/train_bert_demo.py
REPO=DIA-MVP/tinyllama-lora-a40 python scripts/train_llama_lora.py
REPO=DIA-MVP/cifar10-simclr-a40 EPOCHS=10 N_TRAIN=15000 python scripts/train_simclr_cifar.py
```

---

## CPU (3 small models)

```bash
srun -p cpu_b1 --cpus-per-task=8 --mem=32G --time=8:00:00 --pty bash

REPO=DIA-MVP/my-bert-sentiment-cpu python scripts/train_bert_demo.py
REPO=DIA-MVP/tinyllama-lora-cpu python scripts/train_llama_lora.py
REPO=DIA-MVP/mnist-ddpm-cpu MAX_MINUTES=30 python scripts/train_ddpm_mnist.py
```

---

## Ingest

```bash
python scripts/ingest_all.py --reset          # A100: rebuild table
python scripts/ingest_all.py --hardware a40   # add A40 rows
python scripts/ingest_all.py --hardware cpu   # add CPU rows
```

Target: `Nodes: 7 (7 with dia_report)` after A100. Check **Files → `nodes.parquet`**
on the dataset page.

---

## Web dashboard

The UI is a **FastAPI** app serving a static front-end (vis-network lineage graph,
KPI rollups, per-model table). Gradio is no longer used.

### Public Space (read-only)

**[DIA-MVP/dia-dashboard](https://huggingface.co/spaces/DIA-MVP/dia-dashboard)** —
reads `DIA-MVP/dia-state-lab-2026`. It is a **Docker** Space running the FastAPI
app; deploy or update it with:

```bash
hf auth login            # once — token needs WRITE access to the DIA-MVP org
python scripts/deploy_space.py            # vendor package + push Space files
python scripts/deploy_space.py --status   # print build/runtime stage + URL
```

`deploy_space.py` uploads a **light** copy of the Space (no torch/transformers):
it vendors `src/ai_impact_accounting/` to the Space root and pushes `space/Dockerfile`,
`space/app.py`, `space/requirements.txt`, and `space/README.md` (whose `sdk: docker`
frontmatter tells HF to build the `Dockerfile`). After it finishes, HF rebuilds the
image; watch `--status` until `stage=RUNNING`, then check the live API:

```bash
curl -s https://dia-mvp-dia-dashboard.hf.space/api/meta
```

A read-only token deploys nothing — the commit fails with `403 Forbidden`. Use
`--space`/`--dataset` to publish elsewhere (e.g. a `vector-institute/*` Space that
reads its own dataset), with a token that has write access to that owner:

```bash
python scripts/deploy_space.py \
  --space vector-institute/dia-dashboard \
  --dataset vector-institute/dia-state-lab-2026
```

The chosen dataset is rendered into the uploaded `Dockerfile` (`ENV DIA_DATASET`)
and the README link, so one script serves multiple Spaces.

Embed on another page: `?embed=1` (hides header/footer chrome).

### Keeping the dataset up to date

The Space serves whatever is in `state.json`. Two independent, complementary
mechanisms keep that current as the team trains and pushes models:

**1. Real-time webhook (push path) — in the Space.** When the Space has both
secrets set, `space/app.py` exposes an ingest webhook: a content push to a model
repo adds/updates its node in the dataset immediately (shared store, so the live
dashboard reflects it on the next request). Without the secrets the Space stays
read-only.

| Space secret | Purpose |
|--------------|---------|
| `HF_TOKEN` | Write token for the dataset repo |
| `WEBHOOK_SECRET` | Shared secret; must match the Hub webhook config |

Configure a Hub webhook (watch the org / model repos) pointing at:

```
https://<space-subdomain>.hf.space/webhooks/webhooks/ingest
```

Webhooks only fire for repos you own/watch — hence the second mechanism.

**2. Nightly crawl (pull path) — outside the Space.** A Space daemon thread is
unreliable because free Spaces sleep on inactivity, so the periodic backfill runs
as a **GitHub Actions** cron (`.github/workflows/crawl.yml`) instead. It discovers
models two complementary ways:

- **`DIA_ORGS` (recommended)** — index *everything the team publishes* by listing
  every model under each org/user. The org list is small and stable, so it never
  goes stale: any new training shows up automatically, whatever base it derives
  from (or none at all). This is the answer to "a base list is never complete."
- **`DIA_BASES` (optional)** — also catch *third-party* derivatives of your models
  via HF `base_model:` tags (forks published outside your orgs).

Set at least one:

```bash
export DIA_DATASET=DIA-MVP/dia-state-lab-2026
export DIA_ORGS=vector-institute,DIA-MVP                 # who to index
export DIA_BASES=distilbert-base-uncased                 # optional: external forks
export HF_TOKEN=...        # write token
python scripts/crawl.py
```

The workflow runs daily (and on manual dispatch). Configure the repo with a
`HF_TOKEN` **secret** and `DIA_DATASET` / `DIA_ORGS` / `DIA_BASES` / `DIA_SPACE`
**variables** (defaults: the lab dataset; set `DIA_ORGS` and/or `DIA_BASES`).

Set `DIA_SPACE` (e.g. `vector-institute/dia-dashboard`) to have the crawl
**restart the Space** when it finishes — the running Space caches `state.json`
in memory, and a restart reloads the freshly-crawled data. With this, the Space
needs no webhook or write token: the nightly loop is simply **crawl → write
dataset → restart Space**. The `HF_TOKEN` used must have write access to both the
dataset and the Space.

### Local viewer

On a login node, after ingest (or for the public lab dataset without write access):

```bash
export DIA_DATASET=DIA-MVP/dia-state-lab-2026   # required — default is dia-state
export DIA_BASES=distilbert-base-uncased          # default base in the UI
python scripts/view_local.py
```

Open **http://127.0.0.1:7860**. `HF_TOKEN` is optional for read-only public datasets;
required to ingest or refresh private state.

| Variable | Purpose |
|----------|---------|
| `DIA_DATASET` | HF dataset repo with `state.json` (default: lab table) |
| `DIA_BASES` | Default base model id in the UI |
| `DIA_STATE_FILE` | Local `state.json` instead of Hub (stress tests; see below) |
| `PORT` / `HOST` | Bind address (default `7860` / `0.0.0.0`) |

**Shareable views:** use **Copy link** in the UI (encodes base, compare, graph scope,
table filter in the URL).

**Graph:** click a node for details; **Focus family** rolls up a subtree; **Full dataset**
returns from family scope. Large families hide node labels (use the table or hover).

**Imputation** (method-based estimates for models without `dia_report`) is implemented
in the API but **not shown in the UI** yet — reserved for future exploration. Dev-only:
append `&impute=1` to the dashboard URL.

### Synthetic stress test (no Hub)

Generate a fake 100-node lineage and load it locally:

```bash
python scripts/generate_synthetic_state.py --nodes 100
DIA_STATE_FILE=tests/fixtures/synth-100-state.json \
DIA_BASES=SYNTH-LAB/base-model \
python scripts/view_local.py
```

Use this to test graph layout and table performance before scaling real ingest.

### Base models in the UI

The **base-model dropdown** is built from ingested lineage (parents and roots in
`state.json`), not from a fixed list. After ingest, any lineage parent or root
appears automatically. **`DIA_BASES`** only sets which id loads first (default:
`distilbert-base-uncased`).

Use the table below to pick a **rollup root** — the model whose family subtree
you want KPIs and the graph scoped to.

| Demo / branch | Pick in base selector | Training script |
|---------------|----------------------|-----------------|
| BERT finetune | `distilbert-base-uncased` | `train_bert_demo.py` |
| BERT chain (v2, v3, …) | `distilbert-base-uncased` (full chain) or an intermediate repo (e.g. `DIA-MVP/my-bert-sentiment-a40-v2`) for a suffix only | `train_bert_demo.py` |
| TinyLlama LoRA | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `train_llama_lora.py` (default `BASE`) |
| Llama 3.2 LoRA | `meta-llama/Llama-3.2-3B-Instruct` | `train_llama_lora.py` with `BASE=meta-llama/Llama-3.2-3B-Instruct` |
| Phi-3 LoRA | `microsoft/Phi-3-mini-4k-instruct` | `train_llama_lora.py` with `BASE=microsoft/Phi-3-mini-4k-instruct`, `LORA_TARGETS=qkv_proj` |
| Phi-3 chain (v2, v3) | `microsoft/Phi-3-mini-4k-instruct` (full chain) or `DIA-MVP/phi3-mini-lora` (suffix) | `train_llama_lora.py` with `BASE=DIA-MVP/phi3-mini-lora-v2` etc. |
| Gemma 2 LoRA | `google/gemma-2-2b-it` | `train_llama_lora.py` with `BASE=google/gemma-2-2b-it` |
| Qwen LoRA | `Qwen/Qwen2.5-7B` | `train_qwen_lora.py` |
| ResNet fine-tune | `microsoft/resnet-50` | `train_resnet50_cifar.py` |
| Distillation (SST-2) | `distilbert-base-uncased-finetuned-sst-2-english` (teacher) | `distill_sst2.py` |
| Distill chain (v2) | `distilbert-base-uncased-finetuned-sst-2-english` or `DIA-MVP/bert-tiny-sst2-distill` | `distill_sst2.py` with `TEACHER=…` |

**LoRA chains:** the card lists the immediate training parent (e.g. lora from
`DIA-MVP/phi3-mini-lora`) and often the HF `base_model` foundation (e.g.
Phi-3). The graph may show both edges; incremental training is the **lora**
parent. Pick the foundation id to roll up the whole branch, or an intermediate
DIA repo for a subtree.

**Region / hardware variants** (euwest, uswest, a40-v4, etc.) appear under the
same base once ingested — switch base to the foundation or chain root above, then
use the table filter or click nodes in the graph.

### How the rollup works

You pick a **base** model; the dashboard rolls up the base plus all its descendants:

1. **Sum incremental footprints** — each model logs only its own training delta;
   the family total is the subtree sum.
2. **Dedupe the DAG** — a merged/shared model is counted once.
3. **Coverage, not a bare total** — shows *% disclosed*; totals are a lower bound.
4. **Provenance kept separate** — `measured` vs `estimated` vs `imputed`.

Chart: base-vs-derivatives when the base disclosed a report; otherwise a
per-model breakdown (an undisclosed foundation base is *unknown*, not zero).
The same explanation is in the dashboard under **"How the footprint is computed."**

---

## Local checkpoints (gitignored)

`out-bert/`, `out-llama-lora/`, `out-qwen-lora/`, `out-resnet50/`, `out-simclr/`, `out-ddpm/`

---

## Reset lab

`python scripts/reset_lab.py --dry-run` then `--yes --local` deletes **A100 model
repos only** (`MODELS_A100` in `ingest_all.py`), the whole lab dataset
(`dia-state-lab-2026`, including any A40/CPU rows already ingested), and local
`out-*`. **A40 and CPU model repos are not deleted.** Re-train and re-ingest all
hardware tiers you need.

Does not touch public `DIA-MVP/dia-state` unless you pass `--include-legacy`.
