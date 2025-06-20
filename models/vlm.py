import torch
import torch.nn as nn
import torch.nn.functional as F
from models import LM, MP, ViT, VLMConfig


class VLM(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = ViT(cfg)
        self.MP = MP(cfg)
        self.decoder = LM(cfg)

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
            probs = torch.softmax(last_token_embds, dim=-1)
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
    def from_pretrained(cls, cfg: VLMConfig):
        """
        Assumes that the pretrained model is a SigLip model
        """
        model = cls(cfg)
        model.vision_encoder = ViT.from_pretrained(cfg)
        model.decoder = LM.from_pretrained(cfg)
        return model
