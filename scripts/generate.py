import os
import sys

import torch

# TODO: fix this by fixing data module.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.processors import get_image_processor, get_tokenizer
from models import VLM, VLMConfig
from PIL import Image

# TODO: Load pretrained model from HuggingFace Hub
torch.manual_seed(0)
cfg = VLMConfig()
device = torch.device("cuda")
print("Using device:", device)

model = VLM(cfg).to(device)
model.eval()
tokenizer = get_tokenizer(cfg.lm_tokenizer)
image_processor = get_image_processor(cfg.vit_img_size)

text = "What do you see?"
encoded_batch = tokenizer.batch_encode_plus([text], return_tensors="pt")
tokens = encoded_batch["input_ids"].to(device)

print(f"input tokens shape is {tokens.shape}")

image_path = "/home/htkumar/vlms/assets/image.png"
image = Image.open(image_path)
image = image_processor(image)
image = image.unsqueeze(0).to(device)

print(f"image shape is {image.shape}")

gen = model.generate(tokens, image, max_new_tokens=10)
print(f"generated tokens shape is {gen.shape}")

print(f"output is {tokenizer.batch_decode(gen, skip_special_tokens=True)[0]}")
