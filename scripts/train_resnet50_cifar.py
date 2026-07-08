"""Full fine-tune of an ImageNet-pretrained ResNet-50 on CIFAR-100, with DIA ``track()``.

The high-energy demo: it updates all ~25M parameters at 128px resolution, which
keeps the GPU saturated and logs a much larger footprint than the BERT or LoRA
runs. Training is bounded by a wall-clock budget (``MAX_MINUTES``, default 40),
so it fits comfortably in 30-60 min on an Apple Silicon (MPS) laptop regardless
of how fast the device is.

Single GPU / laptop::

    MAX_MINUTES=50 IMG=160 python scripts/train_resnet50_cifar.py

Multi-GPU on one node (default when Slurm allocates 2+ GPUs)::

    python scripts/train_resnet50_cifar.py

Uses ``nn.DataParallel`` — no ``torchrun`` / process-group setup, which avoids
hostname/IPv6 hangs on many HPC clusters. NVML/CodeCarbon still see all GPUs.

Optional DDP (only on clusters where ``torchrun`` works)::

    USE_DDP=1 torchrun --nnodes=1 --nproc_per_node=2 \\
      --rdzv_endpoint=127.0.0.1:29500 scripts/train_resnet50_cifar.py

Requires the ``examples`` extra.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import torch
import torch.distributed as dist
from huggingface_hub import HfApi, get_token
from torch import nn
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.models import ResNet50_Weights, resnet50

from ai_impact_accounting import track

from dia_finalize import exit_from_finalize, finalize_run

if TYPE_CHECKING:
    from ai_impact_accounting.producer.tracking import track as Track

BASE = os.getenv("BASE", "microsoft/resnet-50")  # the pretrained parent (lineage)
REPO = os.getenv("REPO", "DIA-MVP/resnet50-cifar100")
OUT = os.getenv("OUT", "out-resnet50")
IMG = int(os.getenv("IMG", "128"))
BATCH = int(os.getenv("BATCH", "64"))
MAX_MINUTES = float(os.getenv("MAX_MINUTES", "40"))


def _hf_token() -> str | None:
    """Return the Hugging Face write token from env or the CLI login cache."""
    return os.getenv("HF_TOKEN") or get_token()


def _use_ddp() -> bool:
    return os.getenv("USE_DDP", "").lower() in ("1", "true", "yes") and "LOCAL_RANK" in os.environ


def _init_distributed() -> tuple[int, int, torch.device]:
    """Initialize torch.distributed when launched via ``torchrun`` with ``USE_DDP=1``."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, int(os.environ.get("WORLD_SIZE", "1")), torch.device("cuda", local_rank)


def _unwrap(model: nn.Module) -> nn.Module:
    if isinstance(model, (DDP, DataParallel)):
        return model.module
    return model


def _train_loop(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    deadline: float,
    *,
    log: bool,
) -> bool:
    """Run until the wall-clock budget expires. Returns True if interrupted."""
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    interrupted = False
    try:
        model.train()
        epoch = 0
        while time.time() < deadline:
            epoch += 1
            sampler = getattr(loader, "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
            running, seen = 0.0, 0
            for x, y in loader:
                if time.time() >= deadline:
                    break
                x, y = x.to(device), y.to(device)
                loss = loss_fn(model(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item() * x.size(0)
                seen += x.size(0)
            if log:
                print(f"  epoch {epoch}  train loss {running / max(seen, 1):.4f}")
    except KeyboardInterrupt:
        interrupted = True
    return interrupted


def main() -> None:
    """Fine-tune a pretrained ResNet-50 on CIFAR-100 and stamp a ``dia_report``."""
    distributed = False
    rank = 0
    use_dataparallel = False

    if _use_ddp():
        distributed = True
        rank, world_size, device = _init_distributed()
        mode = f"cuda:{device.index} (DDP x{world_size})"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
        use_dataparallel = n_gpu > 1
        mode = f"cuda (DataParallel x{n_gpu})" if use_dataparallel else "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        mode = "mps"
    else:
        device = torch.device("cpu")
        mode = "cpu"

    is_main = rank == 0
    if is_main:
        print(f"Device: {mode}  |  Base: {BASE}  |  {IMG}px  |  budget {MAX_MINUTES:.0f} min")

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
    if is_main:
        CIFAR100(root=os.path.join(OUT, "data"), train=True, download=True)
    if distributed:
        dist.barrier()
    ds = CIFAR100(root=os.path.join(OUT, "data"), train=True, download=False, transform=train_tf)

    sampler = DistributedSampler(ds, shuffle=True) if distributed else None
    workers = int(os.getenv("WORKERS", "4"))
    loader = DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        persistent_workers=workers > 0,
        drop_last=True,
    )

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 100)  # CIFAR-100 head
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[device.index])
    elif use_dataparallel:
        model = DataParallel(model)

    ckpt = os.path.join(OUT, "resnet50_cifar100.pt")
    deadline = time.time() + MAX_MINUTES * 60
    interrupted = False
    tracker: Track | None = None

    track_ctx = track(base_model=BASE, relation="finetune") if is_main else nullcontext()
    with track_ctx as t:
        if is_main:
            tracker = t
        interrupted = _train_loop(model, loader, device, deadline, log=is_main)

    if distributed:
        dist.barrier()

    if not is_main or tracker is None:
        if distributed:
            dist.destroy_process_group()
        return

    token = _hf_token()

    def _save() -> None:
        os.makedirs(OUT, exist_ok=True)
        torch.save(_unwrap(model).state_dict(), ckpt)

    def _push() -> None:
        api = HfApi(token=token)
        api.create_repo(REPO, exist_ok=True)
        print(f"Pushing weights to {REPO} ...")
        api.upload_file(path_or_fileobj=ckpt, path_in_repo="resnet50_cifar100.pt", repo_id=REPO)
        print(f"Pushing DIA report to {REPO} card ...")
        tracker.push(REPO, token=token)

    code = finalize_run(
        tracker,
        out_dir=OUT,
        repo=REPO,
        token=token,
        base_model=BASE,
        interrupted=interrupted,
        save_fn=_save,
        push_fn=_push,
    )
    if distributed:
        dist.destroy_process_group()
    exit_from_finalize(code)


if __name__ == "__main__":
    main()
