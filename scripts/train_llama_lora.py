"""LoRA fine-tune of a small Llama-family LLM, instrumented with DIA ``track()``.

Defaults to TinyLlama-1.1B so it runs on a Mac (MPS) or CPU. To train a real
Llama-3-8B instead, run on a CUDA GPU and set:

    BASE=meta-llama/Llama-3.1-8B  REPO=DIA-MVP/llama31-8b-lora  python train_llama_lora.py

The point of this file: DIA instrumentation is model-agnostic. The only
DIA-specific lines are ``with track(...)`` and ``t.push(...)``; everything else is
a normal PEFT/LoRA training loop. Requires the ``examples`` extra.
"""

import os
import sys

import torch
from datasets import load_dataset
from huggingface_hub import HfFolder
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from ai_impact_accounting import track


BASE = os.getenv("BASE", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
REPO = os.getenv("REPO", "DIA-MVP/tinyllama-lora-demo")
OUT = os.getenv("OUT", "out-llama-lora")
N_EXAMPLES = int(os.getenv("N_EXAMPLES", "1500"))
EPOCHS = int(os.getenv("EPOCHS", "1"))
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def _hf_token() -> str | None:
    """Return the Hugging Face write token from env or the CLI login cache."""
    return os.getenv("HF_TOKEN") or HfFolder.get_token()


def main() -> None:
    """LoRA fine-tune a small Llama-family model and stamp a ``dia_report``."""
    token = _hf_token()
    if not token:
        print("Run: huggingface-cli login   (or export HF_TOKEN=...)")
        sys.exit(1)

    print(f"Device: {DEVICE}  |  Base: {BASE}  |  {N_EXAMPLES} ex x {EPOCHS} epoch(s)")

    tok = AutoTokenizer.from_pretrained(BASE, token=token)
    if tok.pad_token is None:  # Llama tokenizers have no pad token
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE, token=token, torch_dtype=torch.float32)

    # LoRA: train tiny adapter matrices on the attention projections, freeze the
    # rest. ~0.1% of params -> fits on a laptop.
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Small instruction dataset, formatted as plain text for causal LM.
    ds = load_dataset("tatsu-lab/alpaca", split=f"train[:{N_EXAMPLES}]")

    def fmt(ex):
        instr, inp, out = ex["instruction"], ex.get("input", ""), ex["output"]
        prompt = f"### Instruction:\n{instr}\n"
        if inp:
            prompt += f"### Input:\n{inp}\n"
        prompt += f"### Response:\n{out}{tok.eos_token}"
        return {"text": prompt}

    ds = ds.map(fmt, remove_columns=ds.column_names)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=256)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir=OUT,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=25,
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)

    # The only DIA-specific block:
    with track(base_model=BASE, relation="lora", region="local-mac") as t:
        trainer.train()

    print(t.checklist_line())

    trainer.model.save_pretrained(OUT)
    tok.save_pretrained(OUT)

    print(f"Pushing LoRA adapter to {REPO} ...")
    trainer.model.push_to_hub(REPO, token=token, commit_message="TinyLlama LoRA demo")
    tok.push_to_hub(REPO, token=token)

    print(f"Pushing DIA report to {REPO} card ...")
    t.push(REPO, token=token)

    print("Done. Check:", f"https://huggingface.co/{REPO}")
    print(f"Dashboard base model: {BASE}")


if __name__ == "__main__":
    main()
