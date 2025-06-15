from typing import ForwardRef

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.config import VLMConfig


class MP(nn.Module):
    """
    Modality projector which projects image tokens from ViT to the same dimension as text tokens.
    First it uses pixel shuffle to reduce the number of image tokens and then uses a linear layer to project to the same dimension as text tokens.
    """

    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor**2)
        self.output_dim = cfg.lm_hidden_dim
        self.scale_factor = cfg.mp_pixel_shuffle_factor

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def pixel_shuffle(self, x):
        """
        Pixel shuffle to reduce the number of image tokens.
        """
        bsz, seq_len, emb_dim = x.shape
        # 32 here as we have 1024 number of tokens.
        seq_root = int(seq_len**0.5)
        assert seq_root**2 == seq_len, "seq_len must be a perfect square"
        assert (
            seq_root % self.scale_factor == 0
        ), "seq_len must be divisible by scale_factor"
        height = width = seq_root
        x = x.view(bsz, height, width, emb_dim)
        x = x.reshape(
            bsz,
            height // self.scale_factor,
            self.scale_factor,
            width // self.scale_factor,
            self.scale_factor,
            -1,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, height // self.scale_factor * width // self.scale_factor, -1)
        return x

    def forward(self, x):
        """
        x: [B, N, D] where N is the number of image tokens and D is the hidden dimension of ViT.
        """
        x = self.pixel_shuffle(x)
        x = self.proj(x)
        return x
