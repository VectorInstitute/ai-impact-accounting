"""LoRA fine-tune of Qwen2.5-7B (bf16) on instructions, instrumented with DIA ``track()``.

A real-scale run: a 7B model fits comfortably on a single A100-80GB in bf16 (no
4-bit needed), trains LoRA adapters on all linear layers, and produces genuine
*measured* energy/carbon numbers rather than the toy footprints of the BERT demo.
Qwen2.5 is Apache-2.0 and ungated, so it runs without a license request.

Runtime is bounded by a step budget so you can dial the footprint:

    DIA_CI=0.03 MAX_STEPS=1000 python scripts/train_qwen_lora.py
    # bigger run:  EPOCHS=1 N_EXAMPLES=50000 SEQ_LEN=2048 BATCH=4 python scripts/train_qwen_lora.py

Set ``DIA_CI`` to your grid's carbon intensity (Ontario ~0.03 kgCO2/kWh) so the
carbon number is real, not the generic default. Requires the ``examples`` extra.
"""

import os
import sys

import torch
from datasets import load_dataset
from huggingface_hub import get_token
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from ai_impact_accounting import track


BASE = os.getenv("BASE", "Qwen/Qwen2.5-7B")
REPO = os.getenv("REPO", "DIA-MVP/qwen2.5-7b-lora-demo")
OUT = os.getenv("OUT", "out-qwen-lora")
N_EXAMPLES = int(os.getenv("N_EXAMPLES", "20000"))
EPOCHS = int(os.getenv("EPOCHS", "1"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "-1"))  # -1 = run full epochs
SEQ_LEN = int(os.getenv("SEQ_LEN", "1024"))
BATCH = int(os.getenv("BATCH", "2"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "8"))
HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else ("mps" if torch.backends.mps.is_available() else "cpu")


def _hf_token() -> str | None:
    """Return the Hugging Face write token from env or the CLI login cache."""
    return os.getenv("HF_TOKEN") or get_token()


def main() -> None:
    """LoRA fine-tune Qwen2.5-7B on Alpaca and stamp a ``dia_report``."""
    token = _hf_token()
    if not token:
        print("Run: hf auth login   (or export HF_TOKEN=...)")
        sys.exit(1)

    n_gpu = torch.cuda.device_count() if HAS_CUDA else 0
    print(f"Device: {DEVICE} ({n_gpu} GPU)  |  Base: {BASE}  |  seq {SEQ_LEN}, batch {BATCH}x{GRAD_ACCUM}")

    tok = AutoTokenizer.from_pretrained(BASE, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if HAS_CUDA and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE, token=token, torch_dtype=dtype)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # incompatible with gradient checkpointing

    # LoRA on every linear layer (attention + MLP) -- ~0.5% of params trainable.
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("tatsu-lab/alpaca", split=f"train[:{N_EXAMPLES}]")

    def fmt(ex: dict) -> dict:
        instr, inp, out = ex["instruction"], ex.get("input", ""), ex["output"]
        prompt = f"### Instruction:\n{instr}\n"
        if inp:
            prompt += f"### Input:\n{inp}\n"
        prompt += f"### Response:\n{out}{tok.eos_token}"
        return {"text": prompt}

    ds = ds.map(fmt, remove_columns=ds.column_names)

    def tokenize(batch: dict) -> dict:
        return tok(batch["text"], truncation=True, padding="max_length", max_length=SEQ_LEN)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir=OUT,
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        bf16=(dtype == torch.bfloat16),
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)

    # The only DIA-specific block. region/CI come from DIA_REGION/DIA_CI env.
    with track(base_model=BASE, relation="lora") as t:
        trainer.train()

    print(t.checklist_line())

    trainer.model.save_pretrained(OUT)
    tok.save_pretrained(OUT)

    print(f"Pushing LoRA adapter to {REPO} ...")
    trainer.model.push_to_hub(REPO, token=token, commit_message="Qwen2.5-7B LoRA demo")
    tok.push_to_hub(REPO, token=token)

    print(f"Pushing DIA report to {REPO} card ...")
    t.push(REPO, token=token)

    print("Done. Check:", f"https://huggingface.co/{REPO}")
    print(f"Dashboard base model: {BASE}")


if __name__ == "__main__":
    main()
