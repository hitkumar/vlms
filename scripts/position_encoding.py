# shows the importance of including position embeddings in self attention

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

text = "The dog chased another dog"
tokens = tokenizer(text, return_tensors="pt")["input_ids"]
print(f"input tokens shape is {tokens.shape}")

# Map tokens back to text
print(f"Original text: {text}")
print(f"Token IDs: {tokens.squeeze().tolist()}")

# Decode individual tokens
individual_tokens = []
for token_id in tokens.squeeze():
    token_text = tokenizer.decode([token_id], skip_special_tokens=False)
    individual_tokens.append(token_text)
    print(f"Token ID {token_id}: '{token_text}'")

embeddings = model.embed_tokens(tokens)
print(f"input embeddings shape is {embeddings.shape}")

# Create random positional encoding vectors
seq_len = embeddings.shape[1]
emb_dim = embeddings.shape[-1]

# Create position indices for each token
positions = torch.arange(seq_len).unsqueeze(0)  # Shape: [1, seq_len]

# Create random positional encodings for each position
torch.manual_seed(42)  # For reproducibility
pos_encodings = torch.randn(1, seq_len, emb_dim)
print(f"Random positional encodings shape: {pos_encodings.shape}")

# Add positional encodings to token embeddings
embeddings_with_pos = embeddings + pos_encodings
print(f"Embeddings with positional encoding shape: {embeddings_with_pos.shape}")

emb_dim = embeddings_with_pos.shape[-1]
W_q = nn.Linear(emb_dim, emb_dim, bias=False)
W_k = nn.Linear(emb_dim, emb_dim, bias=False)
W_v = nn.Linear(emb_dim, emb_dim, bias=False)

mha = nn.MultiheadAttention(emb_dim, num_heads=4, batch_first=True)

with torch.no_grad():
    for param in mha.parameters():
        nn.init.normal_(param, std=0.1)

# Use embeddings with positional encoding for attention
output, _ = mha(
    W_q(embeddings_with_pos), W_k(embeddings_with_pos), W_v(embeddings_with_pos)
)
dog_1 = output[0, 2]
dog_2 = output[0, 5]
print("\nAfter adding random positional encoding:")
print(f"dog embeddings identical: {torch.allclose(dog_1, dog_2, atol=1e-6)}")

# Compare with original embeddings (without positional encoding)
output_no_pos, _ = mha(W_q(embeddings), W_k(embeddings), W_v(embeddings))
dog_1_no_pos = output_no_pos[0, 2]
dog_2_no_pos = output_no_pos[0, 5]
print("Without positional encoding:")
print(
    f"dog embeddings identical: {torch.allclose(dog_1_no_pos, dog_2_no_pos, atol=1e-6)}"
)
