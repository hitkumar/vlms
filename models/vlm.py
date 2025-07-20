import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # Import get_tokenizer here to avoid circular import
        from data import get_tokenizer

        self.tokenizer = get_tokenizer(
            cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template
        )

    def forward(self, input_ids, image, attention_mask=None, targets=None):
        # Process image to be in the same embedding space as text tokens
        # [B, 64, 576]
        image_embeds = self.vision_encoder(image)
        image_embeds = self.MP(image_embeds)
        batch_size, seq_len, _ = image_embeds.shape

        # [B, N, D]
        token_embds = self.decoder.token_embedding(input_ids)
        # [B, N + 64, D]
        combined_embds = torch.cat((image_embeds, token_embds), dim=1)

        if attention_mask is not None:
            # all image tokens should be attended to
            image_attention_mask = torch.ones(
                (batch_size, seq_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)

        lm_out = self.decoder(combined_embds, attention_mask=attention_mask)
        loss = None
        if targets is not None:
            # shape is [B, N, vocab_size]
            logits = self.decoder.head(lm_out)
            logits = logits[:, image_embeds.size(1) :, :]
            # print(f"logits shape: {logits.shape}, targets shape: {targets.shape}")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        return lm_out, loss

    @torch.no_grad()
    def generate(self, input_ids, image, attention_mask=None, max_new_tokens=20):
        # Process image to be in the same embedding space as text tokens
        # [B, 64, 576]
        image_embeds = self.vision_encoder(image)
        image_embeds = self.MP(image_embeds)
        batch_size, seq_len, _ = image_embeds.shape

        # [B, N, D]
        token_embds = self.decoder.token_embedding(input_ids)
        # [B, N + 64, D]
        combined_embds = torch.cat((image_embeds, token_embds), dim=1)

        if attention_mask is not None:
            # all image tokens should be attended to
            image_attention_mask = torch.ones(
                (batch_size, seq_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)

        outputs = combined_embds
        generated_tokens = torch.zeros(
            (batch_size, max_new_tokens), device=input_ids.device, dtype=input_ids.dtype
        )

        for i in range(max_new_tokens):
            model_out = self.decoder(outputs, attention_mask=attention_mask)
            # [B, emb_dim]
            last_token_embds = model_out[:, -1, :]
            if not self.cfg.lm_use_tokens:
                # [B, emb_dim] -> [B, vocab_size]
                last_token_embds = self.decoder.head(last_token_embds)

            # [B, vocab_size]
            filtered_logits = top_k_top_p_filtering(last_token_embds)
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(-1)

            # [B, 1, emb_dim]
            next_embds = self.decoder.token_embedding(next_token)
            outputs = torch.cat((outputs, next_embds), dim=1)

            if attention_mask is not None:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (batch_size, 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )

        return generated_tokens

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

    def push_to_hub(self, repo_id: str, private: bool = False):
        """
        Push the model and configuration to the Hugging Face Hub.

        Args:
            repo_id (str): The repo ID on the Hugging Face Hub.
        """
        from huggingface_hub import create_repo, upload_folder

        # Create repo
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            print("Saving model to tmp directory: ", save_path)
            # Save to tmp directory
            self.save_pretrained(save_path)

            # Save model card
            # with open(os.path.join(save_path, "README.md"), "w") as f:
            #     f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            # Upload
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload nanoVLM using push_to_hub",
            )
