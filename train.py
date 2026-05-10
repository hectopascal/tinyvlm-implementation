
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tinyvlm.model import TinyVLM
from tinyvlm.data import LLaVAPretrainDataset, collate
from tinyvlm.config import load_config
from tinyvlm.utils import *
from transformers import AutoImageProcessor
from pathlib import Path

def main():
    cfg = load_config("/content/drive/MyDrive/tinyvlm/configs/stage1_pretrain.yaml")

    logger = TBLogger(f"{cfg.out_dir}/tb")
    model = TinyVLM(cfg.model).cuda()
    img_proc = AutoImageProcessor.from_pretrained(cfg.model.vision_model)

    ds = LLaVAPretrainDataset(cfg.data.data_json, cfg.data.image_root, model.tokenizer, img_proc)
    dl = DataLoader(ds,
                    batch_size=cfg.train.batch_size,
                    shuffle=True,
                    num_workers=4,
                    collate_fn=lambda b: collate(b, model.tokenizer.pad_token_id)
                    )

    # Only projector + LoRA params get optimized. Verify this!
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    opt = AdamW(trainable, lr=cfg.train.lr, weight_decay=0.0)
    sched = CosineAnnealingLR(opt, T_max=cfg.train.max_steps)


    model.train()
    step = 0
    for epoch in range(cfg.train.epochs):
        for batch in dl:
            # print(dl)
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
            out.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); opt.zero_grad()

            if step % cfg.log_every == 0:
                logger.log(step,
                **{"train/loss": out.loss.item(),
                    "train/lr": sched.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item()})
            if step % cfg.save_every == 0 and step > 0:
                save_ckpt(model, opt, sched, step, cfg, cfg.out_dir)
            step += 1
            if step >= cfg.train.max_steps:
                break
        if step >= cfg.train.max_steps:
            break

if __name__ == "__main__":
    import sys; main()