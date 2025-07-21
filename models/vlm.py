import json
import os
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from data import get_tokenizer
from models import LM, MP, ViT, VLMConfig
from models.utils import top_k_top_p_filtering

from safetensors.torch import load_model, save_model


class VLM(nn.Module):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()
        self.cfg = cfg
        if load_backbone:
            print("Loading backbone from safetensors...")
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder = LM.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder = LM(cfg)

        self.MP = MP(cfg)
        self.load_backbone = load_backbone

        self.tokenizer = get_tokenizer(
            cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template
        )

    def replace_image_tokens(self, token_embds, input_ids, image_embeds):
        """
        Replace embeddings for image tokens in token_embds with embeddings from image_embeds.

        Args:
            token_embds: Tensor of shape [batch_size, seq_len, hidden_dim] containing token embeddings
            input_ids: Tensor of shape [batch_size, seq_len] containing input token IDs
            image_embeds: Tensor of shape [batch_size, img_seq_len, hidden_dim] containing image embeddings

        Returns:
            Tensor of shape [batch_size, seq_len, hidden_dim] with image token embeddings replaced
        """
        updated_token_embds = token_embds.clone()
        # [B, N]
        mask_image_tokens = input_ids == self.tokenizer.image_token_id
        updated_token_embds[mask_image_tokens] = image_embeds.view(
            -1, image_embeds.size(-1)
        ).to(updated_token_embds.dtype)
        return updated_token_embds

    def forward(self, input_ids, images, attention_mask=None, targets=None):
        if isinstance(images, list):
            if not images:
                raise ValueError("No images provided.")
            else:
                if isinstance(images[0], list):
                    # each image is of dim [1, 3, img_size, img_size]
                    images = [img for img_list in images for img in img_list]
                # [B, 3, W, H]
                images = torch.cat(images, dim=0).to(input_ids.device)
        # Process image to be in the same embedding space as text tokens
        # [B, mp_image_token_length, 576]
        image_embeds = self.vision_encoder(images)
        image_embeds = self.MP(image_embeds)

        # [B, N, D]
        token_embds = self.decoder.token_embedding(input_ids)
        # Replace image tokens with image embeddings
        combined_embds = self.replace_image_tokens(token_embds, input_ids, image_embeds)

        # These are the logits, shape is [B, N, lm_hidden_dim]
        lm_out = self.decoder(combined_embds, attention_mask=attention_mask)
        loss = None
        if targets is not None:
            # shape is [B, N, vocab_size]
            logits = self.decoder.head(lm_out)
            # print(f"logits shape: {logits.shape}, targets shape: {targets.shape}")
            # targets already has -100 for tokens which should be ignored during loss computation
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        return lm_out, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        images,
        attention_mask=None,
        max_new_tokens=20,
        top_k=50,
        top_p=0.9,
        temperature=0.5,
        greedy=False,
    ):
        if isinstance(images, list):
            if not images:
                raise ValueError("No images provided.")
            else:
                if isinstance(images[0], list):
                    # each image is of dim [1, 3, img_size, img_size]
                    images = [img for img_list in images for img in img_list]
                # [B, 3, W, H]
                images = torch.cat(images, dim=0).to(input_ids.device)
        # Process image to be in the same embedding space as text tokens
        # [B, mp_image_token_length, 576]
        image_embeds = self.vision_encoder(images)
        image_embeds = self.MP(image_embeds)

        # [B, N, D]
        token_embds = self.decoder.token_embedding(input_ids)
        # Replace image tokens with image embeddings
        combined_embds = self.replace_image_tokens(token_embds, input_ids, image_embeds)
        batch_size = combined_embds.size(0)

        # These are the logits, shape is [B, N, lm_hidden_dim]
        lm_out = self.decoder(combined_embds, attention_mask=attention_mask)
        if not self.cfg.lm_use_tokens:
            # [B, N, vocab_size]
            logits = self.decoder.head(lm_out)
        logits = logits[:, -1, :]
        newly_generated_ids_list = []

        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(logits)
                probs = torch.softmax(filtered_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            # next_token_id is of dim [B, 1]
            newly_generated_ids_list.append(next_token_id)
            # [B, 1, lm_hidden_dim]
            next_token_embd = self.decoder.token_embedding(next_token_id)
            combined_embds = torch.cat([combined_embds, next_token_embd], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (batch_size, 1),
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        ),
                    ),
                    dim=1,
                )
            lm_out = self.decoder(combined_embds, attention_mask=attention_mask)
            if not self.cfg.lm_use_tokens:
                # [B, N, vocab_size]
                logits = self.decoder.head(lm_out)
            logits = logits[:, -1, :]

        generated_ids = torch.cat(newly_generated_ids_list, dim=1)

        # Truncate the sequences to the first eos token or to the max length
        if self.tokenizer.eos_token_id is not None and generated_ids.numel() > 0:
            seq_len = generated_ids.size(1)
            device = generated_ids.device
            eos_mask = generated_ids == self.tokenizer.eos_token_id
            col_indices_for_min = torch.arange(seq_len, device=device)
            masked_col_indices = torch.where(
                eos_mask,
                col_indices_for_min.unsqueeze(0).expand_as(generated_ids),
                seq_len + 1,
            )
            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values
            actual_first_eos_indices = torch.clamp(
                first_eos_indices_values, max=seq_len
            )
            col_indices_for_comparison = (
                torch.arange(seq_len, device=device)
                .unsqueeze(0)
                .expand_as(generated_ids)
            )
            mask = col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)
            generated_ids[mask] = self.tokenizer.pad_token_id

        # [B, max_new_tokens]
        return generated_ids

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, revision: Optional[str] = None
    ) -> "VLM":
        """
        Loads a pretrained VLM model from a repo_id or path.
        """
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")
            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            if not os.path.exists(weights_path):
                raise ValueError(
                    f"Weights file not found at {weights_path}. Please provide a valid path."
                )
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        with open(config_path, "r") as f:
            cfg = VLMConfig(**json.load(f))

        model = cls(cfg, load_backbone=False)

        # print(get_safetensors_keys(weights_path))

        load_model(model, weights_path)
        return model

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model using safetensors.
        """
        os.makedirs(save_directory, exist_ok=True)
        # save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        # save weights
        save_model(self, os.path.join(save_directory, "model.safetensors"))

    # def push_to_hub(self, repo_id: str, private: bool = False):
    #     """
    #     Push the model and configuration to the Hugging Face Hub.
    #     Don't push to hf hub for now.

    #     Args:
    #         repo_id (str): The repo ID on the Hugging Face Hub.
    #     """
    #     from huggingface_hub import create_repo, upload_folder

    #     # Create repo
    #     repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
    #     repo_id = repo_url.repo_id
    #     print("Created repo: ", repo_url)

    #     with tempfile.TemporaryDirectory() as save_path:
    #         print("Saving model to tmp directory: ", save_path)
    #         # Save to tmp directory
    #         self.save_pretrained(save_path)

    #         # Save model card
    #         # with open(os.path.join(save_path, "README.md"), "w") as f:
    #         #     f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

    #         # Upload
    #         return upload_folder(
    #             repo_id=repo_id,
    #             repo_type="model",
    #             folder_path=save_path,
    #             commit_message="Upload nanoVLM using push_to_hub",
    #         )
