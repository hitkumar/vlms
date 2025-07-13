import argparse
import os
import time

import models.config as config
import models.utils as utils
import torch
import torch.optim as optim
from data import (
    get_image_processor,
    get_tokenizer,
    MMStarCollator,
    MMStarDataset,
    VQACollator,
    VQADataset,
)
from datasets import concatenate_datasets, load_dataset
from models import TrainConfig, VLM, VLMConfig
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def get_run_name(train_cfg: TrainConfig):
    batch_size = f"bs{train_cfg.batch_size}"
    epochs = f"ep{train_cfg.epochs}"
    learning_rate = f"lr{train_cfg.lr_backbones}-{train_cfg.lr_mp}"
    date = time.strftime("%m%d")

    return f"nanoVLM_{batch_size}_{epochs}_{learning_rate}_{date}"


def test_mmstar(model, tokenizer, test_dataloader, device):
    total_samples = 0
    total_correct = 0
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            model_out = model.generate(input_ids, image, attention_mask)

            correct_answeer = tokenizer.batch_decode(labels, skip_special_tokens=True)
            generated_answer = tokenizer.batch_decode(
                model_out, skip_special_tokens=True
            )
            is_correct = utils.check_multiple_choice_with_regex(
                generated_answer, correct_answeer
            )
            total_samples += len(is_correct)
            if is_correct:
                total_correct += sum(is_correct)

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy


def get_dataloaders(train_cfg, vlm_cfg: VLMConfig):
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


def train(train_config: TrainConfig, vlm_config: VLMConfig):
    train_dataloader, test_dataloader = get_dataloaders(train_config, vlm_config)
    tokenizer = get_tokenizer(vlm_config.lm_tokenizer)

    if train_config.resume_from_vlm_checkpoint:
        model = VLM.from_pretrained(vlm_config.vlm_checkpoint_path)
    else:
        model = VLM(vlm_config, load_backbone=vlm_config.vlm_load_backbone_weights)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"nanoVLM model has {num_params/1e6:.2f} million parameters")

    # Define optimizr groups, we are setting higfher LR for the MP as it is being trained from scratch. While decoder and vision encoder are being loaded from pretrained checkpoints.
    param_groups = [
        {"params": model.MP.parameters(), "lr": train_config.lr_mp},
        {
            "params": list(model.vision_encoder.parameters())
            + list(model.decoder.parameters()),
            "lr": train_config.lr_backbones,
        },
    ]
    optimizer = optim.AdamW(param_groups)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if train_config.compile:
        model = torch.compile(model)

    print("Model compiled successfully")

    epoch_times = []
    best_accuracy = 0
    global_step = 0
    best_model_state_dict = None
    for epoch in range(train_config.epochs):
        print(f"Starting epoch {epoch+1}/{train_config.epochs}")
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_tokens_processed = 0

        for batch in train_dataloader:
            batch_start_time = time.time()
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(input_ids, images, attention_mask, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print("forward pass done")
            batch_loss = loss.item()
            # print(f"batch loss at step {global_step} is {batch_loss}")
            total_train_loss += batch_loss  # loss for the entire epoch
            num_tokens = torch.sum(attention_mask).item()
            # we should have 64 tokens per image in default setting
            num_tokens += images.shape[0] * (
                (images.shape[2] / vlm_config.vit_patch_size**2)
                / (vlm_config.mp_pixel_shuffle_factor**2)
            )
            total_tokens_processed += num_tokens
            batch_end_time = time.time()
            # number of seconds, not milliseconds
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = num_tokens / batch_duration

            if train_config.eval_in_epochs and global_step % 100 == 0:
                epoch_eval_accuracy = test_mmstar(
                    model, tokenizer, test_dataloader, device
                )
                if epoch_eval_accuracy > best_accuracy:
                    best_accuracy = epoch_eval_accuracy
                    best_step = global_step
                    best_model_state_dict = getattr(
                        model, "_orig_mod", model
                    ).state_dict()

                print(
                    f"step: {global_step}/{len(train_dataloader) * train_config.epochs}, loss: {batch_loss:.4f}, tokens_per_sec: {tokens_per_second:.2f}, accuracy: {epoch_eval_accuracy:.2f}"
                )
                if vlm_config.hf_repo_name is not None:
                    model.push_to_hub(vlm_config.hf_repo_name)

            global_step += 1

        avg_train_loss = total_train_loss / len(train_dataloader)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        print(
            f"Epoch {epoch+1}/{train_config.epochs}, train_loss: {avg_train_loss:.4f}, tokens_per_sec: {epoch_tokens_per_second:.2f}, epoch duration: {epoch_duration:.2f}"
        )

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    total_training_time = sum(epoch_times)
    total_samples_processed = len(train_dataloader.dataset) * train_config.epochs
    avg_time_per_sample = total_training_time / total_samples_processed
    print(
        f"Average time per sample: {avg_time_per_sample:.2f} seconds, average epoch time: {avg_epoch_time:.2f} seconds"
    )
    accuracy = test_mmstar(model, tokenizer, test_dataloader, device)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_state_dict = getattr(model, "_orig_mod", model).state_dict()
        best_step = global_step

    # save the best model
    if best_model_state_dict is not None:
        torch.save(best_model_state_dict, vlm_config.vlm_checkpoint_path)
        print(
            f"Saved the best model to {vlm_config.vlm_checkpoint_path}, best step is {best_step}, best accuracy is {best_accuracy:.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_mp", type=float, help="learning rate for MP")
    parser.add_argument(
        "--lr_backbones", type=float, help="learning rate for backbones"
    )
    args = parser.parse_args()

    vlm_config = config.VLMConfig()
    train_config = config.TrainConfig()

    if args.lr_mp is not None:
        train_config.lr_mp = args.lr_mp
    if args.lr_backbones is not None:
        train_config.lr_backbones = args.lr_backbones

    train(train_config, vlm_config)


if __name__ == "__main__":
    main()


# Scratch pad

# def main1():
#     vlm_config = config.VLMConfig()
#     train_config = config.TrainConfig()
#     train_dataloader, test_dataloader = get_dataloaders(train_config, vlm_config)
#     tokenizer = get_tokenizer(vlm_config.lm_tokenizer)
#     vit = ViT.from_pretrained(vlm_config)
#     mp = MP(vlm_config)
#     lm = LM.from_pretrained(vlm_config)
#     print(f"lm use_tokens is {lm.lm_use_tokens}, lm tie tokens is {lm.lm_tie_weights}")

#     vlm = VLM.from_pretrained(vlm_config)
#     print(f"vlm cls flag is {vlm.vision_encoder.cls_flag}")

#     print(f"eos token is {tokenizer.eos_token_id}")
#     print("train dataloader")
#     for batch in train_dataloader:
#         print(batch.keys())
#         vlm_out = vlm(
#             batch["input_ids"],
#             batch["image"],
#             batch["attention_mask"],
#             batch["labels"],
#         )
#         print(f"vlm output shape is {vlm_out[0].shape}")
#         print(batch.keys())
#         for k, v in batch.items():
#             print(k, v.shape)
#         break

#     print("test dataloader")
#     for batch in test_dataloader:
#         for k, v in batch.items():
#             print(k, v.shape)
#             if k == "image":
#                 vit_output = vit(v)
#                 mp_output = mp(vit_output)

#                 print(
#                     f"vit output shape is {vit(v).shape}, mp output shape is {mp_output.shape}"
#                 )
#             if k == "labels":
#                 print(
#                     f"Raw value of v[0]: {v[0]}, decoded value: {tokenizer.decode(v[0])}"
#                 )
#         break
