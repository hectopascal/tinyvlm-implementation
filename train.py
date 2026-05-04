import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tinyvlm.model import TinyVLM
from tinyvlm.data import LLaVAPretrainDataset, collate
from tinyvlm.config import load_config

def main(cfg_path):
    cfg = load_config(cfg_path)
    wandb.init(project="tinyvlm", config=cfg.__dict__)

    model = TinyVLM(cfg).cuda()
    img_proc = AutoImageProcessor.from_pretrained(cfg.vision_model)

    ds = LLaVAPretrainDataset(cfg.data_json, cfg.image_root, model.tokenizer, img_proc)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4,
                    collate_fn=lambda b: collate(b, model.tokenizer.pad_token_id))

    # Only projector + LoRA params get optimized. Verify this!
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    opt = AdamW(trainable, lr=cfg.lr, weight_decay=0.0)
    sched = CosineAnnealingLR(opt, T_max=cfg.max_steps)

    model.train()
    step = 0
    for epoch in range(cfg.epochs):
        for batch in dl:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); opt.zero_grad()

            if step % 10 == 0:
                wandb.log({"loss": out.loss.item(), "lr": sched.get_last_lr()[0]}, step=step)
            if step % cfg.save_every == 0 and step > 0:
                save_ckpt(model, cfg.out_dir, step)
            step += 1
            if step >= cfg.max_steps: return

if __name__ == "__main__":
    import sys; main(sys.argv[1])