import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scale


def _apply_lora(
    module: nn.Module,
    rank: int,
    alpha: int,
    dropout: float,
    target_keywords: tuple[str, ...] | None,
    prefix: str = "",
) -> int:
    replaced = 0
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and (target_keywords is None or any(k in full_name.lower() for k in target_keywords)):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced += 1
        else:
            replaced += _apply_lora(child, rank, alpha, dropout, target_keywords, full_name)
    return replaced


class DinoV3VisionEncoder(nn.Module):
    def __init__(
        self,
        in_chans: int,
        feature_chans: int,
        num_pool_layers: int,
        image_width: int,
        output_dim: int,
        model_name: str | None = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        **_: object,
    ) -> None:
        super().__init__()
        if model_name is None:
            model_name = os.environ.get("DINOV3_MODEL_NAME", "facebook/dinov3-vits16plus-pretrain-lvd1689m")

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        local_only = os.environ.get("DINOV3_LOCAL_ONLY") or os.environ.get("HF_HUB_OFFLINE")
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token,
            local_files_only=bool(local_only),
        )
        for param in self.model.parameters():
            param.requires_grad = False

        _apply_lora(
            self.model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_keywords=None,
        )

        self.in_chans = in_chans
        self.image_width = image_width
        self.num_pool_layers = num_pool_layers

        self.patch_size = getattr(self.model.config, "patch_size", 16)
        self.hidden_size = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "embed_dim", None)
        if self.hidden_size is None:
            raise ValueError("Could not determine DINOv3 hidden size from config.")

        self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)

        self.input_proj = nn.Identity()
        if in_chans != 3:
            self.input_proj = nn.Conv2d(in_chans, 3, kernel_size=1, padding=0, bias=False)

        self.encoder_init_chan = feature_chans * (2 ** (num_pool_layers - 1))
        self.token_proj = nn.Linear(self.hidden_size, self.encoder_init_chan, bias=False)

        self.pyramid_projs = nn.ModuleList()
        for i in range(num_pool_layers):
            out_ch = feature_chans * (2**i)
            self.pyramid_projs.append(
                nn.Sequential(
                    nn.Conv2d(self.hidden_size, out_ch, kernel_size=1, padding=0, bias=False),
                    nn.GroupNorm(4, out_ch),
                    nn.SiLU(inplace=True),
                )
            )

    def _split_tokens(
        self,
        tokens: torch.Tensor,
        img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, _ = tokens.shape
        h = img.shape[-2] // self.patch_size
        w = img.shape[-1] // self.patch_size
        num_patches = h * w
        prefix = 1 + self.num_register_tokens
        if num_tokens - prefix != num_patches:
            prefix = 0
            if num_tokens != num_patches:
                raise ValueError(f"Token count {num_tokens} does not match expected patches {num_patches}.")
        patch_tokens = tokens[:, prefix:, :]
        prefix_tokens = tokens[:, :prefix, :] if prefix > 0 else tokens[:, :1, :]
        return prefix_tokens, patch_tokens

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        if x.dim() != 4:
            raise NotImplementedError("x dim should be 4")
        x = self.input_proj(x)
        outputs = self.model(x, return_dict=True)
        tokens = outputs.last_hidden_state

        prefix_tokens, patch_tokens = self._split_tokens(tokens, x)
        batch_size = x.shape[0]
        h = x.shape[-2] // self.patch_size
        w = x.shape[-1] // self.patch_size
        patch_map = patch_tokens.transpose(1, 2).reshape(batch_size, self.hidden_size, h, w)

        projected_tokens = self.token_proj(tokens)
        if prefix_tokens.shape[1] > 0:
            img_feature = projected_tokens[:, 0, :]
        else:
            img_feature = projected_tokens.mean(dim=1)

        stack_feature: list[torch.Tensor] = []
        for i in range(self.num_pool_layers):
            target = self.image_width // (2**i)
            feat = F.interpolate(patch_map, size=(target, target), mode="bilinear", align_corners=False)
            feat = self.pyramid_projs[i](feat)
            stack_feature.append(feat)

        return img_feature, projected_tokens, stack_feature
