"""Knowledge distillation on SST-2, instrumented with DIA ``track()``.

A different training paradigm from the other demos: a small student learns to
match a larger pretrained teacher's softened logits (plus the true labels). It is
the first example to use the ``distill`` lineage relation, so the dashboard shows
the student as a ``distill`` derivative of its teacher.

Teacher and student share the BERT WordPiece vocab, so one tokenizer serves both.
Runs on Apple Silicon (MPS) or CPU. Requires the ``examples`` extra.

    EPOCHS=2 N_EXAMPLES=20000 python scripts/distill_sst2.py
"""

import os
import sys

import torch
from datasets import load_dataset
from huggingface_hub import get_token
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ai_impact_accounting import track


TEACHER = os.getenv("TEACHER", "distilbert-base-uncased-finetuned-sst-2-english")
STUDENT = os.getenv("STUDENT", "google/bert_uncased_L-2_H-128_A-2")  # BERT-tiny
REPO = os.getenv("REPO", "DIA-MVP/bert-tiny-sst2-distill")
OUT = os.getenv("OUT", "out-distill")
N_EXAMPLES = int(os.getenv("N_EXAMPLES", "12000"))
EPOCHS = int(os.getenv("EPOCHS", "2"))
TEMP = float(os.getenv("TEMP", "2.0"))
ALPHA = float(os.getenv("ALPHA", "0.5"))  # weight on the hard-label CE loss
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def _hf_token() -> str | None:
    """Return the Hugging Face write token from env or the CLI login cache."""
    return os.getenv("HF_TOKEN") or get_token()


def main() -> None:
    """Distill a teacher classifier into a small student and stamp a ``dia_report``."""
    token = _hf_token()
    if not token:
        print("Run: hf auth login   (or export HF_TOKEN=...)")
        sys.exit(1)

    print(f"Device: {DEVICE}  |  Teacher: {TEACHER}  |  Student: {STUDENT}")

    tok = AutoTokenizer.from_pretrained(TEACHER, token=token)
    teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER, token=token).to(DEVICE).eval()
    student = AutoModelForSequenceClassification.from_pretrained(STUDENT, num_labels=2, token=token).to(DEVICE)

    raw = load_dataset("nyu-mll/glue", "sst2", split=f"train[:{N_EXAMPLES}]")

    def collate(batch: list[dict]) -> tuple[dict, torch.Tensor]:
        enc = tok([b["sentence"] for b in batch], truncation=True, padding=True, max_length=128, return_tensors="pt")
        labels = torch.tensor([b["label"] for b in batch])
        return enc.to(DEVICE), labels.to(DEVICE)

    loader = DataLoader(raw, batch_size=32, shuffle=True, collate_fn=collate, drop_last=True)
    opt = torch.optim.AdamW(student.parameters(), lr=5e-4)
    ce = torch.nn.CrossEntropyLoss()
    kl = torch.nn.KLDivLoss(reduction="batchmean")

    # The only DIA-specific block: relation="distill" records the teacher as parent.
    with track(base_model=TEACHER, relation="distill") as t:
        student.train()
        for epoch in range(EPOCHS):
            running, seen = 0.0, 0
            for enc, labels in loader:
                with torch.no_grad():
                    teacher_logits = teacher(**enc).logits
                student_logits = student(**enc).logits
                # Soft-target KL (temperature-scaled) + hard-label cross-entropy.
                soft = kl(
                    torch.log_softmax(student_logits / TEMP, dim=-1),
                    torch.softmax(teacher_logits / TEMP, dim=-1),
                ) * (TEMP * TEMP)
                loss = ALPHA * ce(student_logits, labels) + (1 - ALPHA) * soft
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item() * labels.size(0)
                seen += labels.size(0)
            print(f"  epoch {epoch + 1}/{EPOCHS}  distill loss {running / max(seen, 1):.4f}")

    print(t.checklist_line())

    student.save_pretrained(OUT)
    tok.save_pretrained(OUT)

    print(f"Pushing student to {REPO} ...")
    student.push_to_hub(REPO, token=token, commit_message="BERT-tiny SST-2 distilled from teacher")
    tok.push_to_hub(REPO, token=token)

    print(f"Pushing DIA report to {REPO} card ...")
    t.push(REPO, token=token)

    print("Done. Check:", f"https://huggingface.co/{REPO}")
    print(f"Dashboard base model: {TEACHER}")


if __name__ == "__main__":
    main()
