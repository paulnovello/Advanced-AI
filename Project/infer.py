"""Minimal single-sample inference with a trained VLM checkpoint.

Usage:
    python infer.py \
        --checkpoint checkpoints/best_step5000 \
        --image path/to/image.jpg \
        --prompt "What is shown in this image?"
"""

import argparse

import torch
from PIL import Image

from data.processors import get_image_processor, get_image_string, get_tokenizer
from models.vision_language_model import VisionLanguageModel


def build_prompt(question: str, tokenizer, cfg) -> str:
    image_string = get_image_string(
        cfg.projector.image_token_length, cfg.image_token
    )
    messages = [
        {
            "role": "user",
            "content": image_string + question,
        }
    ]
    prompt = tokenizer.apply_chat_template(
        [messages], tokenize=False, add_generation_prompt=True
    )
    if isinstance(prompt, list):
        return prompt[0]
    return prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single inference with a trained VLM checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint directory.")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt / question to ask about the image.")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate.")
    return parser.parse_args()


def main():
    args = parse_args()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    print(f"Loading checkpoint from {args.checkpoint} ...")
    model = VisionLanguageModel.from_pretrained(args.checkpoint).to(device)
    model.eval()

    tokenizer = get_tokenizer(model.cfg.lm.tokenizer, model.cfg.image_token)
    image_processor = get_image_processor(model.cfg.vit.img_size)

    image = Image.open(args.image).convert("RGB")
    pixel_values = image_processor(image).unsqueeze(0).to(device)

    prompt = build_prompt(args.prompt, tokenizer, model.cfg)
    encoded = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids,
            pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            greedy=True,
        )

    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("\n--- VLM Answer ---")
    print(answer)


if __name__ == "__main__":
    main()