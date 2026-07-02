"""Denoising diffusion model (DDPM) trained from scratch on MNIST, with DIA ``track()``.

A generative paradigm distinct from the classifier/LLM demos: it learns to reverse
a noising process with a compact timestep-conditioned CNN denoiser. Trained from
random init, so it is a family root (``base_model="scratch"``). Runs on Apple
Silicon (MPS) or CPU; bounded by a wall-clock budget so it fits a chosen window.

    MAX_MINUTES=30 python scripts/train_ddpm_mnist.py

Requires the ``examples`` extra.
"""

import math
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import torch
from huggingface_hub import HfApi, get_token
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from ai_impact_accounting import track

from dia_finalize import exit_from_finalize, finalize_run


REPO = os.getenv("REPO", "DIA-MVP/mnist-ddpm")
OUT = os.getenv("OUT", "out-ddpm")
BATCH = int(os.getenv("BATCH", "128"))
TIMESTEPS = int(os.getenv("TIMESTEPS", "1000"))
CH = int(os.getenv("CH", "64"))
MAX_MINUTES = float(os.getenv("MAX_MINUTES", "30"))
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def _hf_token() -> str | None:
    """Return the Hugging Face write token from env or the CLI login cache."""
    return os.getenv("HF_TOKEN") or get_token()


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding (as in the original DDPM)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    """A residual conv block that injects the timestep embedding."""

    def __init__(self, ch: int, temb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)
        self.temb = nn.Linear(temb_dim, ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class Denoiser(nn.Module):
    """Compact timestep-conditioned CNN that predicts the noise added to an image."""

    def __init__(self, ch: int = 64, n_blocks: int = 4):
        super().__init__()
        self.temb_dim = ch * 4
        self.temb = nn.Sequential(nn.Linear(ch, self.temb_dim), nn.SiLU(), nn.Linear(self.temb_dim, self.temb_dim))
        self.stem = nn.Conv2d(1, ch, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(ch, self.temb_dim) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))
        self.ch = ch

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.temb(_timestep_embedding(t, self.ch))
        h = self.stem(x)
        for block in self.blocks:
            h = block(h, temb)
        return self.head(h)


def main() -> None:
    """Train a small DDPM on MNIST and stamp a ``dia_report``."""
    token = _hf_token()
    if not token:
        print("Run: hf auth login   (or export HF_TOKEN=...)")
        sys.exit(1)

    print(f"Device: {DEVICE}  |  {TIMESTEPS} steps  |  budget {MAX_MINUTES:.0f} min")

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    ds = MNIST(root=os.path.join(OUT, "data"), train=True, download=True, transform=tf)
    workers = int(os.getenv("WORKERS", "2"))
    loader = DataLoader(
        ds, batch_size=BATCH, shuffle=True, num_workers=workers, persistent_workers=workers > 0, drop_last=True
    )

    # Linear beta schedule -> cumulative products for the closed-form forward process.
    betas = torch.linspace(1e-4, 0.02, TIMESTEPS, device=DEVICE)
    acp = torch.cumprod(1.0 - betas, dim=0)
    sqrt_acp, sqrt_one_minus_acp = acp.sqrt(), (1 - acp).sqrt()

    model = Denoiser(CH).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    ckpt = os.path.join(OUT, "mnist_ddpm.pt")
    interrupted = False
    with track(base_model="scratch", relation="finetune") as t:  # region auto-detected from DIA_REGION/AWS_REGION
        try:
            model.train()
            deadline = time.time() + MAX_MINUTES * 60
            epoch = 0
            while time.time() < deadline:
                epoch += 1
                running, seen = 0.0, 0
                for x, _ in loader:
                    if time.time() >= deadline:
                        break
                    x = x.to(DEVICE)
                    ts = torch.randint(0, TIMESTEPS, (x.size(0),), device=DEVICE)
                    noise = torch.randn_like(x)
                    x_t = sqrt_acp[ts][:, None, None, None] * x + sqrt_one_minus_acp[ts][:, None, None, None] * noise
                    loss = nn.functional.mse_loss(model(x_t, ts), noise)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    running += loss.item() * x.size(0)
                    seen += x.size(0)
                print(f"  epoch {epoch}  noise-MSE {running / max(seen, 1):.4f}")
        except KeyboardInterrupt:
            interrupted = True

    def _save() -> None:
        os.makedirs(OUT, exist_ok=True)
        torch.save(model.state_dict(), ckpt)

    def _push() -> None:
        api = HfApi(token=token)
        api.create_repo(REPO, exist_ok=True)
        print(f"Pushing weights to {REPO} ...")
        api.upload_file(path_or_fileobj=ckpt, path_in_repo="mnist_ddpm.pt", repo_id=REPO)
        print(f"Pushing DIA report to {REPO} card ...")
        t.push(REPO, token=token)

    code = finalize_run(
        t,
        out_dir=OUT,
        repo=REPO,
        token=token,
        base_model="scratch",
        interrupted=interrupted,
        dashboard_hint=REPO,
        save_fn=_save,
        push_fn=_push,
    )
    exit_from_finalize(code)


if __name__ == "__main__":
    main()
