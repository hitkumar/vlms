import torch
import torch.nn as nn
import torch.nn.functional as F
from models.config import VLMConfig

"""
Implement ViT from scratch
- Construct embeddings for each image patch using VitPatchEmbedding
- Construct ViT block using attention and mlp blocks
- Construct Vit using Vit blocks.
"""


class VitPatchEmbedding(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.img_size = cfg.vit_img_size
        self.patch_size = cfg.vit_patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.cls_flag = cfg.vit_cls_flag
        self.emb_dim = cfg.vit_hidden_dim

        # This takes image of size [B, 3, W, H] and outputs [B, emb_dim, num_patches_w, num_patches_h]
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        if self.cls_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
            self.position_embedding = nn.Parameter(
                torch.rand(1, self.num_patches + 1, self.emb_dim)
            )
        else:
            self.position_embedding = nn.Parameter(
                torch.rand(1, self.num_patches, self.emb_dim)
            )

    def forward(self, x):
        # [B, 3, W, H] -> [B, emb_dim, num_patches_w, num_patches_h]
        x = self.conv(x)
        # [B, emb_dim, num_patches_w, num_patches_h] -> [B, emb_dim, num_patches]
        x = x.flatten(2)
        # [B, emb_dim, num_patches] -> [B, num_patches, emb_dim]
        x = x.transpose(1, 2)
        if self.cls_flag:
            # Repeat the cls_token tensor along the batch dimension to match the batch size of x
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + self.position_embedding
        return x


class ViTMultiHeadAttention(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.n_heads = cfg.vit_n_heads
        self.emb_dim = cfg.vit_hidden_dim
        assert (
            self.emb_dim % self.n_heads == 0
        ), "Embedding dimension must be divisible by number of heads"
        self.head_dim = self.emb_dim // self.n_heads

        self.qkv_proj = nn.Linear(self.emb_dim, self.emb_dim * 3, bias=True)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=True)

        self.dropout = cfg.vit_dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("Flash attention not available, using default attention")

    def forward(self, x):
        B, T, N = x.shape
        qkv = self.qkv_proj(x)
        # [B, T, N] shape
        q, k, v = qkv.split(N, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # [B, n_heads, T, head_dim]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # [B, n_heads, T, head_dim]
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # [B, n_heads, T, head_dim]
        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            # [B, n_heads, T, T]
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
            attn = attn.softmax(dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v  # [B, n_heads, T, head_dim]

        y = y.transpose(1, 2).contiguous().reshape(B, T, N)  # [B, T, N]
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y


class VitMLP(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.activation_fn = nn.GELU(approximate="tanh")
        self.fc1 = nn.Linear(cfg.vit_hidden_dim, cfg.vit_inter_dim)
        self.fc2 = nn.Linear(cfg.vit_inter_dim, cfg.vit_hidden_dim)
        self.dropout = nn.Dropout(cfg.vit_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.attn = ViTMultiHeadAttention(cfg)
        self.ln1 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.mlp = VitMLP(cfg)
        self.ln2 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.patch_embedding = VitPatchEmbedding(cfg)
        self.cls_flag = cfg.vit_cls_flag
        self.dropout = nn.Dropout(cfg.vit_dropout)
        self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.vit_n_blocks)])
        self.layer_norm = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        # x dim is [B, 3, W, H] -> [B, num_patches+1, emb_dim]
        x = self.patch_embedding(x)
        for block in self.blocks:
            x = block(x)

        if self.cls_flag:
            # Get the CLS token embedding
            x = self.layer_norm(x[:, 0])
        else:
            # Output shape is [B, num_patches, emb_dim]
            x = self.layer_norm(x)
        return x

    @classmethod
    def from_pretrained(cls, cfg: VLMConfig):
        """
        Assumes that the pretrained model is a SigLip model
        """
        import safetensors
        from huggingface_hub import hf_hub_download
        from transformers import SiglipVisionConfig

        hf_config = SiglipVisionConfig.from_pretrained(cfg.vit_model_type)
        cfg.vit_dropout = hf_config.attention_dropout
        cfg.vit_hidden_dim = hf_config.hidden_size
        cfg.vit_img_size = hf_config.image_size
        cfg.vit_inter_dim = hf_config.intermediate_size
        cfg.vit_ln_eps = hf_config.layer_norm_eps
        cfg.vit_n_heads = hf_config.num_attention_heads
        cfg.vit_n_blocks = hf_config.num_hidden_layers
        cfg.vit_patch_size = hf_config.patch_size

        model = cls(cfg)
        safetensors_file = hf_hub_download(
            repo_id=cfg.vit_model_type, filename="model.safetensors"
        )
        sd = model.state_dict()

        mapping = {
            "vision_model.embeddings.patch_embedding.weight": "patch_embedding.conv.weight",
            "vision_model.embeddings.patch_embedding.bias": "patch_embedding.conv.bias",
            "vision_model.embeddings.position_embedding.weight": "patch_embedding.position_embedding",
            "vision_model.post_layernorm.weight": "layer_norm.weight",
            "vision_model.post_layernorm.bias": "layer_norm.bias",
        }

        for i in range(cfg.vit_n_blocks):
            # Layer norms
            mapping[f"vision_model.encoder.layers.{i}.layer_norm1.weight"] = (
                f"blocks.{i}.ln1.weight"
            )
            mapping[f"vision_model.encoder.layers.{i}.layer_norm1.bias"] = (
                f"blocks.{i}.ln1.bias"
            )
            mapping[f"vision_model.encoder.layers.{i}.layer_norm2.weight"] = (
                f"blocks.{i}.ln2.weight"
            )
            mapping[f"vision_model.encoder.layers.{i}.layer_norm2.bias"] = (
                f"blocks.{i}.ln2.bias"
            )

            # MLP
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc1.weight"] = (
                f"blocks.{i}.mlp.fc1.weight"
            )
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc1.bias"] = (
                f"blocks.{i}.mlp.fc1.bias"
            )
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc2.weight"] = (
                f"blocks.{i}.mlp.fc2.weight"
            )
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc2.bias"] = (
                f"blocks.{i}.mlp.fc2.bias"
            )

            # Output projection
            mapping[f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = (
                f"blocks.{i}.attention.out_proj.weight"
            )
            mapping[f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = (
                f"blocks.{i}.attention.out_proj.bias"
            )

        # Load the pretrained weights
        with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
            pretrained_sd = {key: f.get_tensor(key) for key in f.keys()}

        # Transfer weights using the mapping
        for hf_key, our_key in mapping.items():
            if hf_key in pretrained_sd and our_key in sd:
                tensor = pretrained_sd[hf_key]
                if tensor.shape == sd[our_key].shape:
                    sd[our_key].copy_(tensor)
                else:
                    if "position_embedding" in hf_key:
                        sd[our_key].copy_(tensor.unsqueeze(0))
                    else:
                        print(
                            f"Shape mismatch for {hf_key} and {our_key}. Skipping transfer."
                        )

        # Handle attention weights separately (q, k, v are combined in our implementation)
        for i in range(cfg.vit_n_blocks):
            # Get separate q, k, v weights from pretrained model
            q_weight = pretrained_sd[
                f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
            ]
            k_weight = pretrained_sd[
                f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
            ]
            v_weight = pretrained_sd[
                f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
            ]

            q_bias = pretrained_sd[
                f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
            ]
            k_bias = pretrained_sd[
                f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
            ]
            v_bias = pretrained_sd[
                f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
            ]

            # Combine them for our qkv_proj layer
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

            sd[f"blocks.{i}.attention.qkv_proj.weight"].copy_(qkv_weight)
            sd[f"blocks.{i}.attention.qkv_proj.bias"].copy_(qkv_bias)

        # Load the state dict into the model
        model.load_state_dict(sd)
        num_params = sum(p.numel() for p in model.parameters())
        print(
            f"Loaded pretrained weights from SigLip model {cfg.vit_model_type}, model has {num_params/1e6:.2f} million parameters"
        )
        return model
