import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..transformer import LayerNorm, Transformer


class TextEncoder(nn.Module):
    context_length: int
    vocab_size: int
    transformer: Transformer
    token_embedding: nn.Embedding
    positional_embedding: nn.Parameter
    layernorm_final: LayerNorm
    text_projection: nn.Parameter
    logit_scale: nn.Parameter

    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        pretrained_model_weights: Path | None,
    ) -> None:
        super().__init__()

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, transformer_width)  # 49408 * 512
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))  # 77x512
        self.ln_final = LayerNorm(transformer_width)  # 512, 512

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))  # 512x512
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if pretrained_model_weights is not None:
            self.load_pretrain_weights(os.path.join(os.path.dirname(__file__), pretrained_model_weights))

    def load_pretrain_weights(
        self,
        pretrained_model_weights: Path | None,
    ) -> None:
        if pretrained_model_weights is None:
            return

        if not os.path.exists(pretrained_model_weights):
            raise ValueError("Path not exists")

        with open(pretrained_model_weights, "rb") as opened_file:
            model = torch.jit.load(opened_file, map_location="cpu").eval()

        model_state_dict = {}
        for key in model.state_dict():
            if key.split(".")[0] in [
                "transformer",
                "token_embedding",
                "positional_embedding",
                "ln_final",
                "text_projection",
                "logit_scale",
            ]:
                model_state_dict[key] = model.state_dict()[key]

        self.load_state_dict(model_state_dict, strict=True)

    def build_attention_mask(self) -> torch.Tensor:
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(
        self,
        text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_embedding(text).type(torch.float32)
        x = x + self.positional_embedding.type(torch.float32)

        x = self.transformer(x)
        x = self.ln_final(x).type(torch.float32)
        full_feat = x

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # print(x)
        return x, full_feat
