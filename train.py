import models.config as config
from data.collators import MMStarCollator, VQACollator
from data.datasets import MMStarDataset, VQADataset
from data.processors import get_image_processor, get_tokenizer
from datasets import concatenate_datasets, load_dataset
from models.lm import LM
from models.mp import MP
from models.vit import ViT
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
    vit = ViT.from_pretrained(vlm_config)
    mp = MP(vlm_config)
    lm = LM.from_pretrained(vlm_config)
    print(f"lm use_tokens is {lm.lm_use_tokens}, lm tie tokens is {lm.lm_tie_weights}")

    print(f"eos token is {tokenizer.eos_token_id}")
    print("train dataloader")
    for batch in train_dataloader:
        print(batch.keys())
        for k, v in batch.items():
            print(k, v.shape)
        break

    print("test dataloader")
    i = 0
    for batch in test_dataloader:
        print(batch.keys())
        for k, v in batch.items():
            print(k, v.shape)
            if k == "image":
                vit_output = vit(v)
                mp_output = mp(vit_output)

                print(
                    f"vit output shape is {vit(v).shape}, mp output shape is {mp_output.shape}"
                )
            if k == "labels":
                print(
                    f"Raw value of v[0]: {v[0]}, decoded value: {tokenizer.decode(v[0])}"
                )
        break


if __name__ == "__main__":
    main()
