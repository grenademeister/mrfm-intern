import os
from pathlib import Path

import torch
from torch import nn


class TextQFormer(nn.Module):
    def __init__(self, d_model: int, n_queries: int, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_queries = n_queries
        self.query_tokens = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ln_q": nn.LayerNorm(d_model),
                "ln_q2": nn.LayerNorm(d_model),
                "self_attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "cross_attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "ffn": nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * d_model, d_model),
                ),
            })
            for _ in range(n_layers)
        ])

    def forward(self, token_feats: torch.Tensor, token_mask: torch.Tensor = None) -> torch.Tensor:
        B, L, D = token_feats.shape
        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)

        key_padding_mask = None
        if token_mask is not None:
            key_padding_mask = (token_mask == 0)  # True where PAD

        for layer in self.layers:
            qn = layer["ln_q"](q)
            q_self, _ = layer["self_attn"](query=qn, key=qn, value=qn, need_weights=False)
            q = q + q_self

            qn = layer["ln_q2"](q)
            attn_out, _ = layer["cross_attn"](
                query=qn, key=token_feats, value=token_feats,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )
            q = q + attn_out
            q = q + layer["ffn"](q)

        return q  # (B, Q, D)

class LISTFMTextTokenExtractor(nn.Module):
    """
    Your LIST-FM text_encoder already returns:
      - pooled embedding: (B, D)
      - full token embeddings: (B, L, D)
    We use the full token embeddings for QFormer.
    """
    def __init__(self, text_encoder: nn.Module, freeze: bool = True):
        super().__init__()
        self.text_encoder = text_encoder

        if freeze:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_encoder.eval()

    def forward(self, text_tokens: torch.Tensor):
        token_mask = (text_tokens != 0).long()  # (B, L)
        _, full_features= self.text_encoder(text_tokens)
        return full_features, token_mask  # (B, L, D), (B, L)