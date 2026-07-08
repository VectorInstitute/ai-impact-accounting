"""Minimal DistilBERT SST-2 fine-tune, instrumented with DIA ``track()``.

Trains on a Mac (MPS) or CPU in ~10-15 min, pushes the weights, and writes a
``dia_report`` block into the model card. Requires the ``examples`` extra:

    pip install "ai-impact-accounting[examples,measure]"

The only DIA-specific lines are ``with track(...)`` and ``t.push(...)``.
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import torch
from datasets import load_dataset
from huggingface_hub import get_token
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ai_impact_accounting import track

from dia_finalize import exit_from_finalize, finalize_run


# distilbert: smaller/faster on Mac. For full BERT use "google-bert/bert-base-uncased"
# Override BASE to fine-tune an existing Hub model (multi-hop chain), e.g.:
#   BASE=DIA-MVP/my-bert-sentiment-a40 REPO=DIA-MVP/my-bert-sentiment-a40-v2 ...
BASE = os.getenv("BASE", "distilbert-base-uncased")
REPO = os.getenv("REPO", "DIA-MVP/my-bert-sentiment")
LINEAGE_MODEL = os.getenv("LINEAGE_MODEL", BASE)
OUT = os.getenv("OUT", "out-bert")
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


def _hf_token() -> str | None:
    """Return the Hugging Face write token from env or the CLI login cache."""
    return os.getenv("HF_TOKEN") or get_token()


def main() -> None:
    """Fine-tune DistilBERT on SST-2, push weights, and stamp a ``dia_report``."""
    token = _hf_token()
    if not token:
        print("Run: hf auth login   (or export HF_TOKEN=...)")
        sys.exit(1)

    print(f"Device: {DEVICE}  |  Base: {BASE}")
    ds = load_dataset("nyu-mll/glue", "sst2", split="train[:8000]")
    ds = ds.train_test_split(test_size=0.2, seed=42)

    tok = AutoTokenizer.from_pretrained(BASE, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2, token=token)

    def tokenize(batch):
        return tok(batch["sentence"], truncation=True, padding="max_length", max_length=128)

    ds = ds.map(tokenize, batched=True)
    cols = ["input_ids", "attention_mask", "label"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in cols])

    args = TrainingArguments(
        output_dir=OUT,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"])

    interrupted = False
    with track(base_model=LINEAGE_MODEL, relation="finetune") as t:  # region auto-detected from DIA_REGION/AWS_REGION
        try:
            trainer.train()
        except KeyboardInterrupt:
            interrupted = True

    def _push() -> None:
        print(f"Pushing weights to {REPO} ...")
        trainer.model.push_to_hub(REPO, token=token, commit_message="DistilBERT SST-2 demo fine-tune")
        print(f"Pushing DIA report to {REPO} card ...")
        t.push(REPO, token=token)

    code = finalize_run(
        t,
        out_dir=OUT,
        repo=REPO,
        token=token,
        base_model=LINEAGE_MODEL,
        interrupted=interrupted,
        save_fn=lambda: (trainer.save_model(OUT), tok.save_pretrained(OUT)),
        push_fn=_push,
    )
    exit_from_finalize(code)


if __name__ == "__main__":
    main()
