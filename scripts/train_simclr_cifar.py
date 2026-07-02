"""SimCLR self-supervised pretraining on CIFAR-10, instrumented with DIA ``track()``.

Unlike the BERT and LoRA demos, this wraps ``track()`` around a plain PyTorch
loop and measures a from-scratch pretrain (the root of a model family). SimCLR
has no model parent, so ``base_model="scratch"`` marks the root; query this
model's id as the base in the dashboard to roll up its derivatives.

Defaults run on CPU or Apple Silicon in a few minutes; override via env, e.g.
``EPOCHS=20 N_TRAIN=50000 python scripts/train_simclr_cifar.py``. Requires the
``examples`` extra.
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, get_token
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

from ai_impact_accounting import track

from dia_finalize import exit_from_finalize, finalize_run


REPO = os.getenv("REPO", "DIA-MVP/cifar10-simclr-resnet18")
OUT = os.getenv("OUT", "out-simclr")
EPOCHS = int(os.getenv("EPOCHS", "20"))
N_TRAIN = int(os.getenv("N_TRAIN", "15000"))  # subset for a quick laptop run
BATCH = int(os.getenv("BATCH", "256"))
PROJ_DIM = int(os.getenv("PROJ_DIM", "128"))
TEMP = float(os.getenv("TEMP", "0.5"))
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def _hf_token():
    return os.getenv("HF_TOKEN") or get_token()


class TwoCrop:
    """Return two independently-augmented views of the same image (SimCLR)."""

    def __init__(self, base):
        self.base = base

    def __call__(self, x):
        return self.base(x), self.base(x)


def _augment():
    # Standard SimCLR augmentation pipeline, adapted to 32x32 CIFAR.
    color = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
    )


class SimCLR(nn.Module):
    """ResNet-18 encoder (adapted for 32x32) + a 2-layer projection head."""

    def __init__(self, proj_dim=128):
        super().__init__()
        backbone = resnet18(weights=None)
        # CIFAR tweak: small 3x3 stem, drop the aggressive maxpool.
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        return self.projector(self.encoder(x))


def nt_xent(z1, z2, temperature):
    """Normalized temperature-scaled cross-entropy (the SimCLR contrastive loss)."""
    n = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    # Positive pairs: i <-> i+n.
    targets = torch.arange(n, device=z.device)
    targets = torch.cat([targets + n, targets])
    return F.cross_entropy(sim, targets)


def main():
    token = _hf_token()
    if not token:
        print("Run: hf auth login   (or export HF_TOKEN=...)")
        sys.exit(1)

    print(f"Device: {DEVICE}  |  {N_TRAIN} imgs x {EPOCHS} epoch(s)  |  batch {BATCH}")

    ds = CIFAR10(root=os.path.join(OUT, "data"), train=True, download=True, transform=TwoCrop(_augment()))
    if N_TRAIN < len(ds):
        ds = Subset(ds, range(N_TRAIN))
    workers = int(os.getenv("WORKERS", "2"))
    loader = DataLoader(
        ds, batch_size=BATCH, shuffle=True, num_workers=workers, persistent_workers=workers > 0, drop_last=True
    )

    model = SimCLR(PROJ_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-6)

    ckpt = os.path.join(OUT, "simclr_resnet18_encoder.pt")
    interrupted = False
    with track(base_model="scratch", relation="finetune") as t:  # region auto-detected from DIA_REGION/AWS_REGION
        try:
            model.train()
            for epoch in range(EPOCHS):
                running = 0.0
                for (v1, v2), _ in loader:
                    v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
                    loss = nt_xent(model(v1), model(v2), TEMP)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    running += loss.item()
                print(f"  epoch {epoch + 1}/{EPOCHS}  contrastive loss {running / len(loader):.4f}")
        except KeyboardInterrupt:
            interrupted = True

    def _save() -> None:
        os.makedirs(OUT, exist_ok=True)
        torch.save(model.encoder.state_dict(), ckpt)

    def _push() -> None:
        api = HfApi(token=token)
        api.create_repo(REPO, exist_ok=True)
        print(f"Pushing encoder weights to {REPO} ...")
        api.upload_file(path_or_fileobj=ckpt, path_in_repo="simclr_resnet18_encoder.pt", repo_id=REPO)
        print(f"Pushing DIA report to {REPO} card ...")
        t.push(REPO, token=token)

    code = finalize_run(
        t,
        out_dir=OUT,
        repo=REPO,
        token=token,
        base_model="scratch",
        interrupted=interrupted,
        dashboard_hint=f"{REPO}  (query this id to roll up its derivatives)",
        save_fn=_save,
        push_fn=_push,
    )
    exit_from_finalize(code)


if __name__ == "__main__":
    main()
