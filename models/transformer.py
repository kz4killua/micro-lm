import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TransformerLanguageModel(nn.Module):
    """A transformer-style language model."""

    def __init__(
        self,
        n_vocab: int,
        context_size: int,
        d_model: int,
        n_layers: int,
        h: int,
        d_k: int,
        d_v: int,
        d_ff: int,
    ):
        super().__init__()

        self.d_model = d_model

        self.token_embeddings = nn.Embedding(n_vocab, d_model)
        self.position_encodings = PositionalEncoding(context_size, d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, h, d_k, d_v, d_ff) for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, n_vocab)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embeddings(x) * (self.d_model**0.5)

        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.position_encodings(positions)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = self.layer_norm(x)

        logits = self.output_head(x)

        return logits


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, h: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, h, d_k, d_v)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.multi_head_attention(x)
        x = self.layer_norm_1(x)
        x = x + self.feed_forward(x)
        x = self.layer_norm_2(x)
        return x


class MultiHeadAttention(nn.Module):
    """Masked multi-head self-attention."""

    def __init__(self, d_model: int, h: int, d_k: int, d_v: int):
        super().__init__()

        self.attention_heads = nn.ModuleList(
            [AttentionHead(d_model, d_k, d_v) for _ in range(h)]
        )
        self.W_O = nn.Linear(h * d_v, d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.concat([head(x) for head in self.attention_heads], dim=-1)
        return self.W_O(x)


class AttentionHead(nn.Module):
    """Scaled dot-product attention."""

    def __init__(self, d_model: int, d_k: int, d_v: int):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        _, T, _ = x.shape

        # Compute keys, queries, and values
        K = self.W_K(x)
        Q = self.W_Q(x)
        V = self.W_V(x)

        # Compute dot products for each query with each key
        attention_scores = Q @ torch.transpose(K, -1, -2)
        attention_scores /= self.d_k**0.5

        # Mask future keys for each query
        forbidden = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        attention_scores = attention_scores.masked_fill(forbidden, -torch.inf)

        # Compute the normalized attention pattern
        attention_pattern = F.softmax(attention_scores, -1)

        # Return the weighted sum of values
        return attention_pattern @ V


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, context_size: int, d_model: int):
        super().__init__()

        # Pre-compute positional encodings for all positions
        i = torch.arange((d_model + 1) // 2)
        base = 10_000 ** (2 * i / d_model)

        positions = torch.arange(context_size).reshape(-1, 1)
        sin_term = torch.sin(positions / base)
        cos_term = torch.cos(positions / base)

        encodings = torch.stack((sin_term, cos_term), dim=-1).flatten(-2)
        encodings = encodings[:, :d_model]
        self.encodings = nn.Buffer(encodings)

    def forward(self, positions: Tensor) -> Tensor:
        return self.encodings[positions]
