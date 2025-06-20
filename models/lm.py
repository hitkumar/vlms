import torch
import torch.nn as nn
import torch.nn.functional as F
from models import VLMConfig


class RMSNorm(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.lm_hidden_dim))
        self.eps = cfg.lm_rms_eps

    def forward(self, x):
        irms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * irms * self.weight


# ROPE implementaion
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_embedding(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        assert (
            cfg.lm_hidden_dim % cfg.lm_n_heads == 0
        ), "Hidden dimension must be divisible by number of heads"

        self.dim = cfg.lm_hidden_dim // cfg.lm_n_heads  # dim of each head
        self.base = cfg.lm_re_base
        self.max_seq_len = cfg.lm_max_position_embeddings

        # Standard RoPE implementation - create frequencies for each dimension
        # freq_i = 1 / (base^(2i/dim)) where i is the dimension index
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.original_max_seq_len = cfg.lm_max_position_embeddings
        self.attention_scaling = cfg.lm_attn_scaling

    @torch.no_grad()
    def forward(self, position_ids, unsqueeze_dim=1):
        """
        Args:
            position_ids: [B, N] where N is the number of tokens in the sequence
        """
        batch_size, seq_len = position_ids.shape
        max_seq_len = position_ids.max() + 1
        if max_seq_len > self.original_max_seq_len:
            scaling_factor = max_seq_len / self.original_max_seq_len
            inv_freq = self.inv_freq / scaling_factor
        else:
            inv_freq = self.inv_freq

        # (B*N, )
        flat_position_ids = position_ids.reshape(-1).float()
        # [B*N, dim // 2]
        freqs = flat_position_ids.unsqueeze(-1) * inv_freq.unsqueeze(0)
        freqs = freqs.reshape(batch_size, seq_len, self.dim // 2)

        emb = torch.cat((freqs, freqs), dim=-1)
        cos = torch.cos(emb) * self.attention_scaling
        sin = torch.sin(emb) * self.attention_scaling
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

        return cos, sin


class LMGroupedQueryAttention(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()

        self.n_heads = cfg.lm_n_heads
        self.n_kv_heads = cfg.lm_n_kv_heads
        self.embd_dim = cfg.lm_hidden_dim
        self.dropout = cfg.lm_dropout

        assert (
            self.n_heads % self.n_kv_heads == 0
        ), "n_heads must be divisible by n_kv_heads"
        assert (
            self.embd_dim % self.n_heads == 0
        ), "embd_dim must be divisible by n_heads"

        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim = self.embd_dim // self.n_heads

        self.q_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.k_proj = nn.Linear(
            self.embd_dim, self.head_dim * self.n_kv_heads, bias=False
        )
        self.v_proj = nn.Linear(
            self.embd_dim, self.head_dim * self.n_kv_heads, bias=False
        )
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resi_dropout = nn.Dropout(self.dropout)

        # assumes that this is available on the device
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x, cos, sin, attention_mask=None):
        """
        x is [B, N, D] where N is the number of tokens and D is the hidden dimension of LM.
        """
        B, T, N = x.shape

        # [B, n_heads, T, head_dim]
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # [B, n_kv_heads, T, head_dim]
        k = (
            self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        )  # [B, n_kv_heads, T, head_dim]
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_embedding(q, k, cos, sin)
        k = k.repeat_interleave(self.n_kv_groups, dim=1)
        v = v.repeat_interleave(self.n_kv_groups, dim=1)

        if attention_mask is not None:
            # shape of mask is [B, N] where N is the number of tokens
            # make shape [B, 1, 1, N] so that it can be broadcasted to Q and K
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # value is -inf for padding tokens and 0 for other tokens, this is added to the attn scores.
            attention_mask = (1 - attention_mask) * torch.finfo(x.dtype).min

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().reshape(B, T, N)
        y = self.out_proj(y)
        y = self.resi_dropout(y)
        return y


class LMMLP(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.embed_dim = cfg.lm_hidden_dim
        self.inter_dim = cfg.lm_inter_dim

        self.activation_fn = F.silu
        self.gate_proj = nn.Linear(self.embed_dim, self.inter_dim, bias=False)
        self.up_proj = nn.Linear(self.embed_dim, self.inter_dim, bias=False)
        self.down_proj = nn.Linear(self.inter_dim, self.embed_dim, bias=False)

    def forward(self, x):
        # x is [B, N, D] where N is the number of tokens and D is the hidden dimension of LM.
        gate = self.activation_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LMBlock(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.attn = LMGroupedQueryAttention(cfg)
        self.mlp = LMMLP(cfg)
        self.norm1 = RMSNorm(cfg)
        self.norm2 = RMSNorm(cfg)

    def forward(self, x, cos, sin, attention_mask=None):
        x = x + self.attn(self.norm1(x), cos, sin, attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class LM(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.cfg = cfg
        self.lm_use_tokens = cfg.lm_use_tokens
        self.lm_tie_weights = cfg.lm_tie_weights

        self.token_embedding = nn.Embedding(cfg.lm_vocab_size, cfg.lm_hidden_dim)
        self.rotary_embd = RotaryEmbedding(cfg)
        self.blocks = nn.ModuleList([LMBlock(cfg) for _ in range(cfg.lm_n_blocks)])
        # Final layer norm
        self.norm = RMSNorm(cfg)

        self.head = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)
        if self.lm_tie_weights:
            self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(self, x, attention_mask=None):
        if self.lm_use_tokens:
            x = self.token_embedding(x)

        B, T, N = x.shape
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_embd(position_ids)

        for block in self.blocks:
            x = block(x, cos, sin, attention_mask)

        x = self.norm(x)
        # compute logits if we are using tokens, else return the embeddings
        if self.lm_use_tokens:
            x = self.head(x)
        return x

    @torch.no_grad()
    def generate(self, inputs, max_new_tokens=20):
        # Generate using greedy decoding
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        generated = inputs.clone()
        for _ in range(max_new_tokens):
            outputs = self(generated)
            last_output = outputs[:, -1, :]
            if self.lm_use_tokens:
                next_token = torch.argmax(last_output, dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=-1)
            else:
                next_token_embedding = last_output.unsqueeze(1)
                generated = torch.cat((generated, next_token_embedding), dim=1)

        return generated

    @classmethod
    def from_pretrained(cls, cfg: VLMConfig):
        """
        Assumes that the pretrained model is a SigLip model
        """
        import safetensors
        import torch
        import torch.nn.init as init
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig

        # Load the HuggingFace config
        hf_config = AutoConfig.from_pretrained(cfg.lm_model_type)

        # Store original HF vocab size before we modify it
        original_vocab_size = hf_config.vocab_size
        print(f"Original vocabulary size from pretrained model: {original_vocab_size}")

        # Configure model parameters from HF config
        cfg.lm_hidden_dim = hf_config.hidden_size
        cfg.lm_inter_dim = hf_config.intermediate_size
        cfg.lm_rms_eps = hf_config.rms_norm_eps
        cfg.lm_re_base = hf_config.rope_theta
        cfg.lm_max_position_embeddings = hf_config.max_position_embeddings
        # We're keeping our own vocab size in cfg, but checking it's larger than original
        if hasattr(cfg, "lm_vocab_size"):
            if cfg.lm_vocab_size < original_vocab_size:
                raise ValueError(
                    f"Config vocab size ({cfg.lm_vocab_size}) is smaller than pretrained model vocab size ({original_vocab_size})"
                )
            print(f"Using extended vocabulary size: {cfg.lm_vocab_size}")
        else:
            # If not specified, use the original
            cfg.lm_vocab_size = original_vocab_size
            print(f"Using original vocabulary size: {cfg.lm_vocab_size}")

        cfg.lm_n_heads = hf_config.num_attention_heads
        cfg.lm_n_kv_heads = hf_config.num_key_value_heads
        cfg.lm_dropout = hf_config.attention_dropout
        cfg.lm_n_blocks = hf_config.num_hidden_layers

        # Create our model with potentially larger vocabulary
        model = cls(cfg)
        safetensors_file = hf_hub_download(
            repo_id=cfg.lm_model_type, filename="model.safetensors"
        )

        sd = model.state_dict()

        mapping = {
            "model.embed_tokens.weight": "token_embedding.weight",
            "model.norm.weight": "norm.weight",
        }

        for i in range(cfg.lm_n_blocks):
            layer_prefix = f"model.layers.{i}."
            block_prefix = f"blocks.{i}."

            mapping.update(
                {
                    f"{layer_prefix}self_attn.q_proj.weight": f"{block_prefix}attn.q_proj.weight",
                    f"{layer_prefix}self_attn.k_proj.weight": f"{block_prefix}attn.k_proj.weight",
                    f"{layer_prefix}self_attn.v_proj.weight": f"{block_prefix}attn.v_proj.weight",
                    f"{layer_prefix}self_attn.o_proj.weight": f"{block_prefix}attn.out_proj.weight",
                    f"{layer_prefix}mlp.gate_proj.weight": f"{block_prefix}mlp.gate_proj.weight",
                    f"{layer_prefix}mlp.up_proj.weight": f"{block_prefix}mlp.up_proj.weight",
                    f"{layer_prefix}mlp.down_proj.weight": f"{block_prefix}mlp.down_proj.weight",
                    f"{layer_prefix}input_layernorm.weight": f"{block_prefix}norm1.weight",
                    f"{layer_prefix}post_attention_layernorm.weight": f"{block_prefix}norm2.weight",
                }
            )

        # Special handling for token embeddings with extended vocabulary
        has_extended_embeddings = False
        with safetensors.safe_open(
            filename=safetensors_file, framework="pt", device="cpu"
        ) as f:
            for hf_key, our_key in mapping.items():
                if hf_key in f.keys() and our_key in sd:
                    tensor = f.get_tensor(hf_key)

                    # Special handling for token embeddings if vocab sizes differ
                    if (
                        hf_key == "model.embed_tokens.weight"
                        and tensor.shape[0] != sd[our_key].shape[0]
                    ):
                        has_extended_embeddings = True
                        print(
                            f"Extending token embeddings from {tensor.shape} to {sd[our_key].shape}"
                        )

                        # Copy existing embeddings to the beginning of our larger embedding matrix
                        sd[our_key][: tensor.shape[0]].copy_(tensor)

                        # Initialize the new embeddings using the same approach as the original model
                        std = 0.02  # Common value, but you might want to adjust based on model
                        init.normal_(sd[our_key][tensor.shape[0] :], mean=0.0, std=std)

                        print(
                            f"Initialized {sd[our_key].shape[0] - tensor.shape[0]} new token embeddings"
                        )
                        sd["head.weight"].copy_(
                            sd[our_key]
                        )  # Update the head weights as well
                    elif tensor.shape == sd[our_key].shape:
                        sd[our_key].copy_(tensor)
                    else:
                        print(
                            f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}"
                        )
                else:
                    if hf_key not in f.keys():
                        print(f"Warning: Key {hf_key} not found in safetensors file")
                    if our_key not in sd:
                        print(f"Warning: Key {our_key} not found in model state dict")

        # Load the state dict
        model.load_state_dict(sd)

        # Handle output projection / language modeling head
        if has_extended_embeddings and hasattr(model, "head") and "head.weight" in sd:
            # If we have a separate output projection layer and extended the vocab
            # we should handle it similarly to the input embeddings
            with safetensors.safe_open(
                filename=safetensors_file, framework="pt", device="cpu"
            ) as f:
                if "lm_head.weight" in f.keys():
                    lm_head = f.get_tensor("lm_head.weight")
                    if lm_head.shape[0] != sd["head.weight"].shape[0]:
                        print(
                            f"Extending LM head from {lm_head.shape} to {sd['head.weight'].shape}"
                        )
                        # Copy existing weights
                        sd["head.weight"][: lm_head.shape[0]].copy_(lm_head)
                        # Initialize new weights
                        std = 0.02
                        init.normal_(
                            sd["head.weight"][lm_head.shape[0] :], mean=0.0, std=std
                        )
                        # Load updated weights
                        model.load_state_dict(sd)

        # Handle weight tying (if needed)
        if (
            cfg.lm_tie_weights
            and hasattr(model, "head")
            and hasattr(model, "token_embedding")
        ):
            model.head.weight = model.token_embedding.weight
            print("Tied token embedding and LM head weights")

        num_params = sum(p.numel() for p in model.parameters())
        print(
            f"Successfully loaded {cfg.lm_model_type} weights from safetensors. model has {num_params/1e6:.2f} million parameters"
        )
        return model
