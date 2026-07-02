"""LoRA fine-tune of a small Llama-family LLM, instrumented with DIA ``track()``.

Defaults to TinyLlama-1.1B so it runs on a Mac (MPS) or CPU. Override the base
model and push target with ``BASE`` and ``REPO`` (no code changes needed). Meta
Llama checkpoints are gated — accept the license on Hugging Face and use a token
with access before running.

Examples (CUDA GPU recommended for real Llama runs):

    # Llama 3.2 3B — good default on a single A100
    BASE=meta-llama/Llama-3.2-3B-Instruct \\
    REPO=DIA-MVP/llama32-3b-lora \\
    N_EXAMPLES=5000 EPOCHS=1 \\
    python scripts/train_llama_lora.py

    # Llama 3.1 8B — larger / higher-quality derivative
    BASE=meta-llama/Llama-3.1-8B-Instruct \\
    REPO=DIA-MVP/llama31-8b-lora \\
    python scripts/train_llama_lora.py

After training, ingest and view in the dashboard:

    DIA_BASES=meta-llama/Llama-3.2-3B-Instruct \\
    ./scripts/run_local.sh DIA-MVP/llama32-3b-lora

The point of this file: DIA instrumentation is model-agnostic. The only
DIA-specific lines are ``with track(...)`` and ``t.push(...)``; everything else is
a normal PEFT/LoRA training loop. Requires the ``examples`` extra.
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

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

from dia_finalize import exit_from_finalize, finalize_run


BASE = os.getenv("BASE", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
REPO = os.getenv("REPO", "DIA-MVP/tinyllama-lora-demo")
OUT = os.getenv("OUT", "out-llama-lora")
N_EXAMPLES = int(os.getenv("N_EXAMPLES", "1500"))
EPOCHS = int(os.getenv("EPOCHS", "1"))
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def _hf_token() -> str | None:
    """Return the Hugging Face write token from env or the CLI login cache."""
    return os.getenv("HF_TOKEN") or get_token()


def main() -> None:
    """LoRA fine-tune a small Llama-family model and stamp a ``dia_report``."""
    token = _hf_token()
    if not token:
        print("Run: hf auth login   (or export HF_TOKEN=...)")
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

    interrupted = False
    with track(base_model=BASE, relation="lora") as t:  # region auto-detected from DIA_REGION/AWS_REGION
        try:
            trainer.train()
        except KeyboardInterrupt:
            interrupted = True

    def _push() -> None:
        print(f"Pushing LoRA adapter to {REPO} ...")
        trainer.model.push_to_hub(REPO, token=token, commit_message="TinyLlama LoRA demo")
        tok.push_to_hub(REPO, token=token)
        print(f"Pushing DIA report to {REPO} card ...")
        t.push(REPO, token=token)

    code = finalize_run(
        t,
        out_dir=OUT,
        repo=REPO,
        token=token,
        base_model=BASE,
        interrupted=interrupted,
        save_fn=lambda: (trainer.model.save_pretrained(OUT), tok.save_pretrained(OUT)),
        push_fn=_push,
    )
    exit_from_finalize(code)


if __name__ == "__main__":
    main()
