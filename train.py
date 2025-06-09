import models.config as config
from data.collators import MMStarCollator, VQACollator
from data.datasets import MMStarDataset, VQADataset
from data.processors import get_image_processor, get_tokenizer
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader


def get_dataloaders(train_cfg, vlm_cfg):
    image_processor = get_image_processor(vlm_cfg.vit_img_size)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)
    combined_training_data = []
    for dataset_name in train_cfg.train_dataset_name:
        train_ds = load_dataset(train_cfg.train_dataset_path, dataset_name)
        combined_training_data.append(train_ds["train"])

    train_ds = concatenate_datasets(combined_training_data)
    test_ds = load_dataset(train_cfg.test_dataset_path)
    print(
        f"train dataset size: {len(train_ds)}, test dataset size: {len(test_ds['val'])}"
    )

    train_dataset = VQADataset(train_ds, tokenizer, image_processor)
    test_dataset = MMStarDataset(test_ds["val"], tokenizer, image_processor)
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    mmstar_collator = MMStarCollator(tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=vqa_collator,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=train_cfg.mmstar_batch_size,
        collate_fn=mmstar_collator,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader


def main():
    vlm_config = config.VLMConfig()
    train_config = config.TrainConfig()
    train_dataloader, test_dataloader = get_dataloaders(train_config, vlm_config)
    tokenizer = get_tokenizer(vlm_config.lm_tokenizer)
    print(f"eos token is {tokenizer.eos_token_id}")
    for batch in train_dataloader:
        print(batch)
        break

    for batch in test_dataloader:
        print(batch)
        break


if __name__ == "__main__":
    main()
