"""Full fine-tune of an ImageNet-pretrained ResNet-50 on CIFAR-100, with DIA ``track()``.

The high-energy demo: it updates all ~25M parameters at 128px resolution, which
keeps the GPU saturated and logs a much larger footprint than the BERT or LoRA
runs. Training is bounded by a wall-clock budget (``MAX_MINUTES``, default 40),
so it fits comfortably in 30-60 min on an Apple Silicon (MPS) laptop regardless
of how fast the device is.

    MAX_MINUTES=50 IMG=160 python scripts/train_resnet50_cifar.py

Requires the ``examples`` extra.
"""

import os
import sys
import time

import torch
from huggingface_hub import HfApi, HfFolder
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.models import ResNet50_Weights, resnet50

from ai_impact_accounting import track


BASE = os.getenv("BASE", "microsoft/resnet-50")  # the pretrained parent (lineage)
REPO = os.getenv("REPO", "DIA-MVP/resnet50-cifar100")
OUT = os.getenv("OUT", "out-resnet50")
IMG = int(os.getenv("IMG", "128"))
BATCH = int(os.getenv("BATCH", "64"))
MAX_MINUTES = float(os.getenv("MAX_MINUTES", "40"))
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def _hf_token() -> str | None:
    """Return the Hugging Face write token from env or the CLI login cache."""
    return os.getenv("HF_TOKEN") or HfFolder.get_token()


def main() -> None:
    """Fine-tune a pretrained ResNet-50 on CIFAR-100 and stamp a ``dia_report``."""
    token = _hf_token()
    if not token:
        print("Run: huggingface-cli login   (or export HF_TOKEN=...)")
        sys.exit(1)

    print(f"Device: {DEVICE}  |  Base: {BASE}  |  {IMG}px  |  budget {MAX_MINUTES:.0f} min")

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.Resize(IMG),
            transforms.RandomCrop(IMG, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ]
    )
    ds = CIFAR100(root=os.path.join(OUT, "data"), train=True, download=True, transform=train_tf)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4, drop_last=True)

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 100)  # CIFAR-100 head
    model = model.to(DEVICE)

    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    # The only DIA-specific block: wrap the training loop. The wall-clock budget
    # (not a fixed epoch count) is what determines the energy this run reports.
    with track(base_model=BASE, relation="finetune") as t:
        model.train()
        deadline = time.time() + MAX_MINUTES * 60
        epoch = 0
        while time.time() < deadline:
            epoch += 1
            running, seen = 0.0, 0
            for x, y in loader:
                if time.time() >= deadline:
                    break
                x, y = x.to(DEVICE), y.to(DEVICE)
                loss = loss_fn(model(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item() * x.size(0)
                seen += x.size(0)
            print(f"  epoch {epoch}  train loss {running / max(seen, 1):.4f}")

    print(t.checklist_line())

    os.makedirs(OUT, exist_ok=True)
    ckpt = os.path.join(OUT, "resnet50_cifar100.pt")
    torch.save(model.state_dict(), ckpt)

    api = HfApi(token=token)
    api.create_repo(REPO, exist_ok=True)
    print(f"Pushing weights to {REPO} ...")
    api.upload_file(path_or_fileobj=ckpt, path_in_repo="resnet50_cifar100.pt", repo_id=REPO)

    print(f"Pushing DIA report to {REPO} card ...")
    t.push(REPO, token=token)

    print("Done. Check:", f"https://huggingface.co/{REPO}")
    print(f"Dashboard base model: {BASE}")


if __name__ == "__main__":
    main()
