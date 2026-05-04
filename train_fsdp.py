import time
from collections import deque
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoImageProcessor

from tinyvlm.config import load_config
from tinyvlm.model import TinyVLM
from tinyvlm.data import LLaVAPretrainDataset, collate
from tinyvlm.utils import print_rank0, get_world_size, resolve_dtype

from tinyvlm.utils import resolve_dtype

def build_param_groups(model, projector_lr, lm_lr):
    """Different LRs for projector vs LoRA params. Required because they
    initialize from very different scales."""
    projector_params, lora_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("projector."):
            projector_params.append(p)
        elif "lora_" in name:
            lora_params.append(p)
        else:
            # Could be embedding table from resize_token_embeddings — assign to lm_lr
            lora_params.append(p)
    return [
        {"params": projector_params, "lr": projector_lr, "name": "projector"},
        {"params": lora_params, "lr": lm_lr, "name": "lm_lora"},
    ]


def main(cfg_path):
    cfg = load_config(cfg_path)
    set_seed(cfg.train.seed)

    dtype = resolve_dtype(cfg.train.dtype)
    # Accelerator handles FSDP wrap, mixed precision, device placement.
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.grad_accum_steps,
        mixed_precision=cfg.train.dtype,
    )
    world_size = accelerator.num_processes
    print_rank0(f"World size: {world_size}, mixed precision: {cfg.train.dtype}")

    # Build model on CPU; accelerator will shard it across GPUs during prepare()
    model = TinyVLM(cfg.model, dtype=dtype)
    img_proc = AutoImageProcessor.from_pretrained(cfg.model.vision_model)

    ds = LLaVAPretrainDataset(
        cfg.data.data_json, cfg.data.image_root,
        model.tokenizer, img_proc,
        max_len=cfg.data.max_seq_len,
        max_samples=cfg.data.max_samples,
    )
    dl = DataLoader(
        ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=lambda b: collate(b, model.tokenizer.pad_token_id),
    )

    param_groups = build_param_groups(model, cfg.train.lr, cfg.train.lm_lr)
    optimizer = AdamW(param_groups, weight_decay=cfg.train.weight_decay,
                      betas=cfg.train.betas)

    # The magic call. accelerator wraps the model in FSDP, distributes the
    # dataloader, sets up the optimizer for sharded params.
    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

    # Throughput tracking
    throughput_window = deque(maxlen=50)
    step = 0
    measured_start = None

    model.train()
    for batch in dl:
        # gradient_accumulation context handles the accum boundary
        with accelerator.accumulate(model):
            t0 = time.perf_counter()
            out = model(**batch)
            accelerator.backward(out.loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Sync to get accurate timing — without this, time.perf_counter
            # captures kernel launch time, not execution time
            accelerator.wait_for_everyone()
            dt = time.perf_counter() - t0

        # Skip warmup steps for measurement
        if step >= cfg.train.warmup_steps:
            if measured_start is None:
                measured_start = time.perf_counter()
            # Tokens this step across all ranks
            local_tokens = batch["input_ids"].numel()
            global_tokens = local_tokens * world_size
            throughput_window.append(global_tokens / dt)

        if step % cfg.train.log_every == 0 and accelerator.is_main_process:
            avg_tps = sum(throughput_window) / len(throughput_window) if throughput_window else 0
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"step {step:4d} | loss {out.loss.item():.3f} | "
                  f"tokens/s {avg_tps:>8.0f} | peak mem {mem_gb:.1f} GB")

        step += 1
        if step >= cfg.train.max_steps:
            break

    # Final measurement
    if accelerator.is_main_process and throughput_window:
        final_tps = sum(throughput_window) / len(throughput_window)
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print_rank0("\n=== RESULTS ===")
        print_rank0(f"World size: {world_size}")
        print_rank0(f"Per-GPU batch: {cfg.train.batch_size}, accum: {cfg.train.grad_accum_steps}")
        print_rank0(f"Effective batch: {cfg.train.batch_size * cfg.train.grad_accum_steps * world_size}")
        print_rank0(f"Avg tokens/sec (last 50 steps): {final_tps:.0f}")
        print_rank0(f"Peak GPU memory: {peak_mem:.2f} GB")

        # Append to results file for plotting
        results_path = Path("fsdp_study/results.csv")
        write_header = not results_path.exists()
        with open(results_path, "a") as f:
            if write_header:
                f.write("world_size,per_gpu_batch,grad_accum,effective_batch,tokens_per_sec,peak_mem_gb,activation_ckpt\n")
            f.write(f"{world_size},{cfg.train.batch_size},{cfg.train.grad_accum_steps},"
                    f"{cfg.train.batch_size * cfg.train.grad_accum_steps * world_size},"
                    f"{final_tps:.0f},{peak_mem:.2f},{cfg.activation_ckpt}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)