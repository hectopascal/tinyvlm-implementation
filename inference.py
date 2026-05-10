
import torch
from PIL import Image
from transformers import AutoImageProcessor
from tinyvlm.model import TinyVLM
from tinyvlm.config import load_config
from tinyvlm.utils import find_latest_ckpt, load_ckpt
from IPython.display import Image as DisplayImage, display


@torch.no_grad()
def generate_caption(model, tokenizer, img_proc, image_path, prompt, max_new_tokens=64):
    model.eval()
    device = next(model.parameters()).device
    full_prompt = f"USER: <image>\n{prompt} ASSISTANT:"
    enc = tokenizer(full_prompt, return_tensors="pt").to(device)
    input_ids = enc.input_ids 

    
    attention_mask = enc.attention_mask
    image = Image.open(image_path).convert("RGB")
    pixel_values = img_proc(image, return_tensors="pt")["pixel_values"].to(device)

    text_embeds = model.lm.get_input_embeddings()(input_ids)
    img_embeds = model.encode_images(pixel_values)
    # dummy labels for inference
    dummy_labels = torch.zeros_like(input_ids)
    merged_embeds, merged_mask, _ = model._splice(
        text_embeds, attention_mask, dummy_labels, img_embeds, input_ids
    )
    out_ids = model.lm.generate(
        inputs_embeds=merged_embeds,
        attention_mask=merged_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,       
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


def main():
    cfg = load_config("configs/stage1_pretrain.yaml")
    model = TinyVLM(cfg.model).to(dtype=torch.bfloat16,device="cuda")
    img_proc = AutoImageProcessor.from_pretrained(cfg.model.vision_model)

    # Load latest checkpoint
    ckpt_path = find_latest_ckpt(cfg.out_dir)
    load_ckpt(ckpt_path, model)

    # Test images
    test_cases = [
        ("/content/000000039769.jpg",       "Describe this image."),
        ("/content/Boeing-Business-Jet-1024x576.jpg",    "What do you see?"),
    ]
    for path, prompt in test_cases:
        print(f"--- {path} ---")
        # Display the image
        display(DisplayImage(filename=path))
        out = generate_caption(model, model.tokenizer, img_proc, path, prompt)
        print(f"Prompt: {prompt}")
        print(f"Output: {out}\n")



if __name__ == "__main__":
    import sys; main()