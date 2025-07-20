import torch
from data import get_image_processor, get_tokenizer, VQACollator, VQADataset
from datasets import concatenate_datasets, load_dataset
from models import TrainConfig, VLMConfig
from torch.utils.data import DataLoader


vlm_config = VLMConfig()
train_cfg = TrainConfig()

tokenizer = get_tokenizer(
    vlm_config.lm_tokenizer,
    extra_special_tokens=vlm_config.vlm_extra_tokens,
    chat_template=vlm_config.lm_chat_template,
)
# print(tokenizer)
random_str = "test str"
random_str_chat_templated = tokenizer.apply_chat_template(
    [{"role": "assistant", "content": random_str}],
    tokenize=True,
    add_special_tokens=False,
    return_dict=True,
)
# random_str_location = random_str_chat_templated.find(random_str)
# print(random_str_chat_templated)
image_processor = get_image_processor(vlm_config.vit_img_size)
tokenizer = get_tokenizer(
    vlm_config.lm_tokenizer, vlm_config.vlm_extra_tokens, vlm_config.lm_chat_template
)

# Load and combine all training datasets
combined_train_data = []
dataset_names_to_load = train_cfg.train_dataset_name

for dataset_name in dataset_names_to_load:
    train_ds = load_dataset(train_cfg.train_dataset_path, dataset_name)
    train_ds["train"][0]  # Check if the dataset is loaded correctly
    combined_train_data.append(train_ds["train"])

train_ds = concatenate_datasets(combined_train_data)
print(f"train dataset size: {len(train_ds)}")

train_dataset = VQADataset(
    train_ds, tokenizer, image_processor, vlm_config.mp_image_token_length
)
sample = train_dataset[0]
print(f"sample keys are: {sample.keys()}")
for key, value in sample.items():
    if isinstance(value, torch.Tensor):
        print(f"{key}: {value.shape}")
    elif isinstance(value, list) and len(value) > 0:
        print(
            f"{key}: list of {len(value)} with first element shape {value[0].shape if hasattr(value[0], 'shape') else type(value[0])}"
        )
    else:
        print(f"{key}: {type(value)}")

vqa_collator = VQACollator(tokenizer, vlm_config.lm_max_length)
train_loader = DataLoader(
    train_dataset,
    batch_size=train_cfg.batch_size,
    collate_fn=vqa_collator,
    num_workers=8,
)
for batch in train_loader:
    print(f"batch keys are: {batch.keys()}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        elif isinstance(value, list) and len(value) > 0:
            print(
                f"{key}: list of {len(value)} with first element shape {value[0][0].shape if hasattr(value[0][0], 'shape') else type(value[0])}"
            )
        else:
            print(f"{key}: {type(value)}")
    break
