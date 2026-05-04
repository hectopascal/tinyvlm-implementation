# tinyvlm/data.py
import io
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

class LLaVAPretrainDataset(Dataset):
    """Expects LLaVA-Pretrain JSON: list of {image, conversations: [{from, value}]}."""
    def __init__(self, json_path, image_root, tokenizer, image_processor, max_len=2048):
        self.data = json.load(open(json_path))
        self.image_root = Path(image_root)
        self.tok = tokenizer
        self.img_proc = image_processor
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        try:
            img = Image.open(self.image_root / ex["image"]).convert("RGB")
        except (FileNotFoundError, OSError):
            # Fall back to next index. Not ideal but keeps training going.
            return self.__getitem__((i + 1) % len(self.data))
            
        pixel_values = self.img_proc(img, return_tensors="pt")["pixel_values"][0]

        # Stage 1 = caption pretraining. Conversation is [human: <image>\n, gpt: caption]
        human = ex["conversations"][0]["value"].replace("<image>", "<image>\n")
        gpt = ex["conversations"][1]["value"]
        prompt = f"USER: {human}\nASSISTANT:"
        full = f"{prompt} {gpt}{self.tok.eos_token}"

        enc = self.tok(full, max_length=self.max_len, truncation=True, return_tensors="pt")
        input_ids = enc.input_ids[0]
        labels = input_ids.clone()
        # Mask the prompt portion — only train on assistant response
        prompt_len = len(self.tok(prompt).input_ids)
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": enc.attention_mask[0],
            "pixel_values": pixel_values,
        }


def collate(batch, pad_id):
    # Pad input_ids/labels/mask to max length in batch. Stack pixel_values.
    # standard padding logic
    max_seqlen = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention_mask = [], [], []
    for x in batch:
      n_pad = max_seqlen - len(x["input_ids"])
      input_ids.append(
        torch.cat(
          [x["input_ids"], torch.full((n_pad,), pad_id, dtype=torch.long)]
          )
        )
      labels.append(
        torch.cat(
          [x["labels"],torch.full((n_pad,), -100, dtype=torch.long)]
        )
      )
      attention_mask.append(
        torch.cat(
          [x["attention_mask"], torch.zeros(n_pad, dtype=torch.long)]
          )
        )

    return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        }
    # max_seq = len()

    
class LLaVAHFDataset(Dataset):
    """Wraps a HF cached dataset. Works whether images are embedded or paths."""
    def __init__(self, hf_repo, split, tokenizer, image_processor,
                 max_seq_len=512, max_samples=None, image_root=None):
        self.ds = load_dataset(hf_repo, split=split)  # uses cache
        if max_samples:
            self.ds = self.ds.select(range(min(max_samples, len(self.ds))))
        self.tok = tokenizer
        self.img_proc = image_processor
        self.max_seq_len = max_seq_len
        self.image_root = image_root  # only used if images are paths

    def __len__(self):
        return len(self.ds)

    def _load_image(self, image_field):
        # Three cases worth handling:
        if image_field is None:
            return None
        if hasattr(image_field, "convert"):
            return image_field.convert("RGB")            # PIL.Image already
        if isinstance(image_field, dict) and "bytes" in image_field:
            return Image.open(io.BytesIO(image_field["bytes"])).convert("RGB")
        if isinstance(image_field, str):
            from pathlib import Path
            return Image.open(Path(self.image_root) / image_field).convert("RGB")
        raise ValueError(f"Unknown image format: {type(image_field)}")

    def __getitem__(self, i):
        ex = self.ds[i]
        img = self._load_image(ex["image"])
        if img is None:
            # Return next valid sample. Crude but fine for a weekend.
            return self.__getitem__((i + 1) % len(self))
        pixel_values = self.img_proc(img, return_tensors="pt")["pixel_values"][0]

        convs = ex["conversations"]
        human = convs[0]["value"]
        if "<image>" not in human:
            human = "<image>\n" + human
        gpt = convs[1]["value"]
        prompt = f"USER: {human}\nASSISTANT:"
        full = f"{prompt} {gpt}{self.tok.eos_token}"

        enc = self.tok(full, max_length=self.max_seq_len, truncation=True, return_tensors="pt")
        input_ids = enc.input_ids[0]
        labels = input_ids.clone()
        prompt_len = len(self.tok(prompt).input_ids)
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": enc.attention_mask[0],
            "pixel_values": pixel_values,
        }