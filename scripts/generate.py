import argparse

import torch

from data import get_image_processor, get_tokenizer
from models import VLM
from PIL import Image

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


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
        default="Explain the scene in detail",
        help="text prompt to feed to the model",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
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
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer, model.cfg.vlm_extra_tokens)
    image_processor = get_image_processor(model.cfg.vit_img_size)
    messages = [
        {
            "role": "user",
            "content": tokenizer.image_token * model.cfg.mp_image_token_length
            + args.prompt,
        }
    ]
    encoded_prompt = tokenizer.apply_chat_template(
        [messages], tokenize=True, add_generation_prompt=True
    )
    tokens = torch.tensor(encoded_prompt).to(device)
    print(
        f"input tokens shape is {tokens.shape}, decoded prompt is {tokenizer.decode(encoded_prompt[0])}"
    )

    image = Image.open(args.image).convert("RGB")
    image_tensor = image_processor(image).unsqueeze(0).to(device)
    print(f"image shape is {image_tensor.shape}")
    for i in range(args.generations):
        gen = model.generate(tokens, image_tensor, max_new_tokens=args.max_new_tokens)
        # print(f"generated tokens shape is {gen.shape}")
        print(
            f"output {i} is {tokenizer.batch_decode(gen, skip_special_tokens=True)[0]}"
        )


if __name__ == "__main__":
    main()
