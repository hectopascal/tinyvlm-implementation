# tinyvlm/config.py
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    # Backbones
    vision_model: str = "google/siglip2-base-patch16-224"
    lm_model: str = "Qwen/Qwen2.5-0.5B"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "v_proj")

    # Projector
    projector_type: str = "mlp2x_gelu"  # leave room for resampler/qformer later

    # What to freeze. Stage 1: freeze vision + LM, train projector only.
    # Stage 2: freeze vision, train projector + LoRA on LM.
    freeze_vision: bool = True
    freeze_lm: bool = True       # True for stage 1, False (LoRA-trainable) for stage 2
    train_projector: bool = True


@dataclass
class DataConfig:
    data_json: str = "/tmp/data/llava_pretrain_subset.json"
    image_root: str = "/tmp/data/images"
    max_seq_len: int = 2048
    image_size: int = 224
    num_workers: int = 4

    # Subsetting for fast iteration
    max_samples: Optional[int] = None  # None = use all


@dataclass
class TrainConfig:
    # Optimizer
    lr: float = 1e-3              # projector-only stage 1 likes high LR
    lm_lr: float = 2e-5           # used in stage 2 when LM is unfrozen
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03

    # Schedule
    epochs: int = 1
    max_steps: int = 5000         # stage 1 ~2-5k steps is plenty for a subset
    batch_size: int = 8
    grad_accum_steps: int = 1     # effective batch = batch_size * grad_accum * world_size

    # Precision
    dtype: str = "bf16"           # "bf16" | "fp16" | "fp32"; auto-downgraded on T4

    # Logging / checkpointing
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000
    keep_last_n_ckpts: int = 2

    # Reproducibility
    seed: int = 42


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Run metadata
    run_name: str = "tinyvlm_stage1"
    out_dir: str = "outputs/tinyvlm_stage1"

    # Switches
    dev_mode: bool = False        # tiny stand-in models, CPU-friendly
    resume: bool = True           # auto-resume from latest ckpt in out_dir
    activation_ckpt: bool = True 

# ---- helpers ----

def load_config(path: str) -> Config:
    """Load YAML, deep-merge into dataclass defaults."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config()
    for section, values in raw.items():
        if hasattr(cfg, section) and hasattr(getattr(cfg, section), "__dataclass_fields__"):
            sub = getattr(cfg, section)
            for k, v in values.items():
                setattr(sub, k, v)
        else:
            setattr(cfg, section, values)
    if cfg.dev_mode:
        _apply_dev_overrides(cfg)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def _apply_dev_overrides(cfg: Config) -> None:
    """Tiny stand-ins so the whole pipeline runs on CPU in seconds."""
    cfg.model.vision_model = "hf-internal-testing/tiny-random-CLIPVisionModel"
    cfg.model.lm_model = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    cfg.model.lora_r = 4
    cfg.data.max_samples = 32
    cfg.data.num_workers = 0
    cfg.train.batch_size = 2
    cfg.train.max_steps = 20
    cfg.train.dtype = "fp32"
    cfg.train.log_every = 1


def save_config(cfg: Config, path: str) -> None:
    """Snapshot the resolved config alongside checkpoints."""
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)