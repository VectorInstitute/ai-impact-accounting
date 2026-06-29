# DIA lab workflow (DIA-MVP)

Train demo models on **A100 / A40 / CPU**, push `dia_report` to Hugging Face model
repos, ingest into **[DIA-MVP/dia-state-lab-2026](https://huggingface.co/datasets/DIA-MVP/dia-state-lab-2026)**,
and view in the Gradio dashboard.

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

## Gradio

On a login node, after ingest:

```bash
export DIA_DATASET=DIA-MVP/dia-state-lab-2026   # required — default is dia-state
export DIA_BASES=distilbert-base-uncased          # pick a row from the table below
python scripts/view_local.py
```

| Family | Base model in UI |
|--------|------------------|
| BERT | `distilbert-base-uncased` |
| TinyLlama LoRA | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Llama 3.2 LoRA | `meta-llama/Llama-3.2-3B-Instruct` |
| Qwen LoRA | `Qwen/Qwen2.5-7B` |
| ResNet | `microsoft/resnet-50` |
| SimCLR | your SimCLR repo id (e.g. `DIA-MVP/cifar10-simclr-a100` or `…-a40`) |
| DDPM | your DDPM repo id (e.g. `DIA-MVP/mnist-ddpm-a100` or `…-cpu`) |

SimCLR and DDPM train from scratch — the dashboard base is the **model repo id**,
not a Hugging Face foundation checkpoint.

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
