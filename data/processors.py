import torchvision.transforms as transforms
from transformers import AutoTokenizer

TOKENIZER_CACHE = {}


def get_image_processor(img_size):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )


def get_tokenizer(name, extra_special_tokens=None):
    if name not in TOKENIZER_CACHE:
        tokenizer_init_kwargs = {"use_fast": True}
        if extra_special_tokens is not None:
            tokenizer_init_kwargs["additional_special_tokens"] = extra_special_tokens
        tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_init_kwargs)
        # for simplicity, we set this to be the eos token.
        tokenizer.pad_token = tokenizer.eos_token
        TOKENIZER_CACHE[name] = tokenizer
    return TOKENIZER_CACHE[name]
