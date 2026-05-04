from torch.utils.tensorboard import SummaryWriter
import json, time
from pathlib import Path
# tinyvlm/utils.py
import os
import random
import shutil
from pathlib import Path
from typing import Optional
import numpy as np
import torch

import torch.distributed as dist

def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1

def print_rank0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


# ---- reproducibility ----

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Don't enable deterministic algorithms — too slow for training.


# ---- precision ----

def resolve_dtype(dtype_str: str) -> torch.dtype:
    """Resolve config string to torch dtype, auto-downgrading bf16 on pre-Ampere."""
    if dtype_str == "fp32":
        return torch.float32
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major < 8:  # T4, V100, etc.
                print("[utils] bf16 requested but GPU is pre-Ampere; using fp16.")
                return torch.float16
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype_str}")


# ---- checkpointing ----

def save_ckpt(model, optimizer, scheduler, step: int, cfg, out_dir: str) -> Path:
    """Save trainable params only — frozen vision tower is huge and pointless to save."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step_{step:07d}.pt"

    # Only save what's trainable. Vision tower weights are on HF Hub already.
    trainable_state = {
        k: v.cpu() for k, v in model.state_dict().items()
        if any(p.requires_grad for n, p in model.named_parameters() if n == k)
    }
    # Cleaner alternative — save by name match:
    trainable_state = {
        k: v.cpu() for k, v in model.state_dict().items()
        if k.startswith("projector.") or "lora_" in k
    }

    torch.save({
        "step": step,
        "model": trainable_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)
    print(f"[utils] saved {path}")
    _prune_old_ckpts(out_dir, cfg.train.keep_last_n_ckpts)
    return path


def find_latest_ckpt(out_dir: str) -> Optional[Path]:
    ckpts = sorted(Path(out_dir).glob("step_*.pt"))
    return ckpts[-1] if ckpts else None


def load_ckpt(path: Path, model, optimizer=None, scheduler=None) -> int:
    """Load checkpoint, return step. strict=False because we only saved trainable params."""
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    # missing keys are fine (frozen vision tower wasn't saved); unexpected keys are a bug
    if unexpected:
        print(f"[utils] WARNING: unexpected keys in ckpt: {unexpected}")
    if optimizer and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    print(f"[utils] resumed from {path} at step {state['step']}")
    return state["step"]


def _prune_old_ckpts(out_dir: Path, keep_last_n: int) -> None:
    ckpts = sorted(out_dir.glob("step_*.pt"))
    for old in ckpts[:-keep_last_n]:
        old.unlink()


# ---- introspection ----

def count_params(model) -> tuple[int, int]:
    """Returns (trainable, total) param counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def print_trainable_params(model) -> None:
    """Sanity check — print which params are actually trainable.
    First thing to check when loss isn't moving."""
    print("[utils] trainable parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {name:60s} {tuple(p.shape)}  {p.numel():>10,}")
    trainable, total = count_params(model)
    print(f"[utils] {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")


# ---- learning rate ----

def get_lr_schedule(optimizer, warmup_steps: int, max_steps: int):
    """Linear warmup then cosine decay to 10% of peak LR."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
    
class TBLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.start = time.time()

    def log(self, step: int, **metrics):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)
        # Print too — useful when TB isn't open
        msg = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in metrics.items())
        print(f"step {step:>6d} | {msg}")

    def log_text(self, tag: str, text: str, step: int):
        self.writer.add_text(tag, text, step)

    def close(self):
        self.writer.close()



class JSONLLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a")
        self.start = time.time()

    def log(self, step, **metrics):
        record = {"step": step, "elapsed": time.time() - self.start, **metrics}
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()
        # Also print so you see it in Colab output
        msg = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in metrics.items())
        print(f"step {step:>6d} | {msg}")

    def close(self):
        self.f.close()