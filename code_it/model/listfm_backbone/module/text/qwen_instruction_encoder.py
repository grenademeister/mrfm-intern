from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM

class QwenInstructionEncoder(nn.Module):
    def __init__(
        self,
        model_path: str,
        lora_path: str | None,
        embed_dim: int,
        trainable: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.trainable = trainable
        self._has_lora = False
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
        if lora_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError("peft is required to load LoRA adapters") from exc
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self._has_lora = True
        hidden_size = self.model.config.hidden_size
        self.proj = nn.Linear(hidden_size, embed_dim, bias=False)

        if not self.trainable:
            if self._has_lora:
                # Freeze base model only, keep LoRA adapters trainable
                for name, param in self.model.named_parameters():
                    if "lora" not in name.lower():
                        param.requires_grad_(False)
                    else:
                        param.requires_grad_(True)
            else:
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad_(False)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs.hidden_states[-1]
        if hidden.dtype != self.proj.weight.dtype:
            hidden = hidden.to(self.proj.weight.dtype)
        full_feat = self.proj(hidden)

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
            pooled = full_feat[torch.arange(full_feat.shape[0], device=full_feat.device), lengths]
        else:
            pooled = full_feat[:, -1]

        return pooled, full_feat

    def train(self, mode: bool = True) -> "QwenInstructionEncoder":
        super().train(mode)
        if not self.trainable and not self._has_lora:
            self.model.eval()
        return self
