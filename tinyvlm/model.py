import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

IMAGE_TOKEN = "<image>"

class MLPProjector(nn.Module):
    """Maps vision features -> LM embedding space. The 'connector' in LLaVA."""
    def __init__(self, vision_dim: int, lm_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, lm_dim),
            nn.GELU(),
            nn.Linear(lm_dim, lm_dim),
        )

    def forward(self, x):  # x: (B, N_patches, vision_dim)
        return self.net(x)


class TinyVLM(nn.Module):
    def __init__(self, cfg, dtype=torch.bfloat16):
        super().__init__()
        # Vision: SigLIP-2, frozen
        self.vision = AutoModel.from_pretrained(cfg.vision_model).vision_model
        for p in self.vision.parameters():
            p.requires_grad = False

        # LM: small causal LM, LoRA-adapted
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.lm_model)
        self.tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        lm = AutoModelForCausalLM.from_pretrained(cfg.lm_model, torch_dtype=dtype)
        lm.resize_token_embeddings(len(self.tokenizer))
        lora_cfg = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        )
        self.lm = get_peft_model(lm, lora_cfg)
        
        # Projector: vision_dim -> lm_dim. Trainable.
        vision_dim = self.vision.config.hidden_size
        lm_dim = self.lm.config.hidden_size
        self.projector = MLPProjector(vision_dim, lm_dim)
        self.to(dtype)

    def encode_images(self, pixel_values):
        with torch.no_grad():
            out = self.vision(pixel_values=pixel_values)
        feats = out.last_hidden_state  # (B, N, D_vision)
        return self.projector(feats)   # (B, N, D_lm)

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        # Get text embeddings, then splice image embeddings at <image> positions.
        text_embeds = self.lm.get_input_embeddings()(input_ids)  # (B, T, D)
        img_embeds = self.encode_images(pixel_values)            # (B, N, D)

        # Build the merged sequence: replace each <image> token with N image embeddings.
        merged_embeds, merged_mask, merged_labels = self._splice(
            text_embeds, attention_mask, labels, img_embeds, input_ids
        )
        return self.lm(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            labels=merged_labels,
            use_cache=False,
        )

    def _splice(self, text_embeds, attn_mask, labels, img_embeds, input_ids):
        # Left as an exercise — but the gist: for each row, find the <image> position,
        # split text_embeds around it, concat [pre, img_embeds, post]. 
        # Same for mask and labels (use -100 for image positions so loss is text-only).
        # See LLaVA's prepare_inputs_labels_for_multimodal for the canonical impl.
        # print(text_embeds.size(), attn_mask.size(), labels.size(), img_embeds.size(), input_ids.size())
        # text embed = B, seqlen, 896
        # mask = B, seqlen
        # labels = B, seqlen
        # img emb = B, 196, 896.  (--> (224/16 )^2 = 14 ^2 = 196)
        # input ids = B, seqlen
        text_im, label_im, mask_im = [],[],[]
        for eg_idx, input_id in enumerate(input_ids):
            img_token_idx = torch.where(input_id==self.image_token_id)[0].item()
            img = img_embeds[eg_idx]
            label = labels[eg_idx]
            label_im.append(torch.cat([
              label[:img_token_idx],
              torch.full((img.shape[0],),-100, device=labels.device, dtype=labels.dtype),
              label[img_token_idx+1:]
            ]))
            mask = attn_mask[eg_idx]
            mask_im.append(torch.cat([
              mask[:img_token_idx],
              torch.full((img.shape[0],),1, device=attn_mask.device, dtype=attn_mask.dtype),
              mask[img_token_idx+1:]
            ]))
            text = text_embeds[eg_idx]
            text_im.append(torch.cat([
              text[:img_token_idx],
              img,
              text[img_token_idx+1:]
            ]))
            # print(text_im[0].size(),label_im[0].size(),mask_im[0].size())
        return torch.stack(text_im), torch.stack(mask_im), torch.stack(label_im)

        # raise NotImplementedError("write this — it's where you'll learn the most")