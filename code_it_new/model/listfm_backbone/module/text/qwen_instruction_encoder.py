import torch
from torch import nn
from transformers import AutoModel


class QwenInstructionEncoder(nn.Module):
    context_length: int
    pad_token_id: int

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        proj_dim: int = 512,
        context_length: int = 64,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.proj = nn.Linear(hidden_size, proj_dim)

        pad_token_id = getattr(self.encoder.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.encoder.config, "eos_token_id", 0)
        self.pad_token_id = int(pad_token_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be a 2D (B, L) tensor.")

        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        attention_mask = attention_mask.to(device=input_ids.device)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        full = outputs.last_hidden_state

        mask = attention_mask.to(dtype=full.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (full * mask).sum(dim=1) / denom

        full = self.proj(full)
        pooled = self.proj(pooled)
        return pooled, full
