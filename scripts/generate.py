import argparse

import torch

from data import get_image_processor, get_tokenizer
from models import VLM
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from VLM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="path to a local .pth checkpoint (if provided). If omitted, model is loaded from HuggingFace Hub.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="/home/htkumar/vlms/assets/image.png",
        help="path to an image file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What do you see?",
        help="text prompt to feed to the model",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="maximum number of tokens to generate",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="number of generations to produce",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = "lusxvr/nanoVLM-450M"

    model = VLM.from_pretrained(checkpoint_path).to(device)
    print("Model loaded successfully")
    model.eval()
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)

    text = args.prompt
    template = f"Question: {text} Answer:"
    encoded = tokenizer.batch_encode_plus([template], return_tensors="pt")
    tokens = encoded["input_ids"].to(device)
    print(f"input tokens shape is {tokens.shape}")

    image = Image.open(args.image)
    image_tensor = image_processor(image).unsqueeze(0).to(device)
    print(f"image shape is {image_tensor.shape}")

    gen = model.generate(tokens, image_tensor, max_new_tokens=10)
    print(f"generated tokens shape is {gen.shape}")

    print(f"output is {tokenizer.batch_decode(gen, skip_special_tokens=True)[0]}")


if __name__ == "__main__":
    main()
