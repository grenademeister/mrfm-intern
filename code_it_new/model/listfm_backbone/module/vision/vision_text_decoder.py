import math

import torch
from torch import nn
from torch.nn import functional as F


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_scale_shift = nn.Linear(cond_dim, dim * 2)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        x = self.norm(x)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim, dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_norm = AdaLayerNorm(dim, dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp_norm = AdaLayerNorm(dim, dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,
        cond_tokens: torch.Tensor,
        cond_vec: torch.Tensor,
    ) -> torch.Tensor:
        h = self.attn_norm(x, cond_vec)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + h

        h = self.cross_norm(x, cond_vec)
        h, _ = self.cross_attn(h, cond_tokens, cond_tokens, need_weights=False)
        x = x + h

        h = self.mlp_norm(x, cond_vec)
        x = x + self.mlp(h)
        return x


class VisionTextDecoder(nn.Module):
    def __init__(
        self,
        out_chans: int = 2,
        feature_chans: int = 64,
        decoder_feature_chans: int | None = None,
        num_pool_layers: int = 5,
        image_width: int = 512,
        block_type: object | None = None,
        instruction_dim: int | None = None,
        input_chans: int | None = None,
    ) -> None:
        super().__init__()
        if decoder_feature_chans is None:
            decoder_feature_chans = feature_chans

        self.out_chans = out_chans
        self.input_chans = input_chans if input_chans is not None else out_chans
        self.image_width = image_width
        self.patch_size = 16

        if image_width % self.patch_size != 0:
            raise ValueError("image_width must be divisible by patch_size")

        self.token_dim = feature_chans * (2 ** (num_pool_layers - 1))
        self.model_dim = min(decoder_feature_chans * 32, 512)
        if self.model_dim % 8 != 0:
            self.model_dim = max(256, decoder_feature_chans * 16)

        self.num_heads = max(4, self.model_dim // 64)
        self.depth = 12
        self.mlp_ratio = 4.0

        self.patch_embed = nn.Conv2d(self.input_chans, self.model_dim, kernel_size=self.patch_size, stride=self.patch_size)
        num_patches = (image_width // self.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.model_dim))

        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.model_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.model_dim, self.model_dim),
        )

        self.instruction_proj = None
        if instruction_dim is not None:
            self.instruction_proj = nn.Linear(instruction_dim, self.model_dim)

        self.cond_proj = nn.Linear(self.token_dim, self.model_dim)

        self.blocks = nn.ModuleList(
            [DiTBlock(self.model_dim, self.num_heads, self.mlp_ratio) for _ in range(self.depth)]
        )

        self.norm_out = nn.LayerNorm(self.model_dim)
        self.head = nn.Linear(self.model_dim, self.out_chans * self.patch_size * self.patch_size)

    def _prepare_instruction(
        self,
        instruction: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if instruction is None:
            return None
        if instruction.dim() == 3:
            instruction = instruction.mean(dim=1)
        if instruction.dim() != 2:
            raise ValueError("instruction must be 2D or 3D tensor.")
        return instruction

    def _prepare_flow_t(
        self,
        flow_t: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if flow_t.dim() == 4:
            flow_t = flow_t.view(batch_size, -1)
        if flow_t.dim() != 2 or flow_t.shape[1] != 1:
            raise ValueError("flow_t must be a (B, 1) or (B, 1, 1, 1) tensor.")
        return flow_t

    def _pad_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.token_dim:
            return x
        if x.shape[-1] < self.token_dim:
            pad = torch.zeros(
                size=(x.shape[0], x.shape[1], self.token_dim - x.shape[-1]),
                device=x.device,
                dtype=x.dtype,
            )
            return torch.cat([x, pad], dim=-1)
        return x[:, :, : self.token_dim]

    def forward(
        self,
        x: torch.Tensor,
        stack_feat: list[torch.Tensor],
        instruction: torch.Tensor = None,
        flow_xt: torch.Tensor | None = None,
        flow_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise NotImplementedError("x dim should be 3")
        if flow_xt is None or flow_t is None:
            raise ValueError("flow_xt and flow_t must be provided for rectified-flow decoding.")

        batch_size = x.shape[0]
        instruction = self._prepare_instruction(instruction)

        flow_xt = F.interpolate(flow_xt, size=(self.image_width, self.image_width), mode="bilinear", align_corners=False)
        xt_tokens = self.patch_embed(flow_xt).flatten(2).transpose(1, 2)
        xt_tokens = xt_tokens + self.pos_embed

        flow_t = self._prepare_flow_t(flow_t, batch_size)
        cond_vec = self.time_mlp(flow_t)

        if instruction is not None:
            if self.instruction_proj is None:
                raise RuntimeError("instruction_dim must be set when instruction is provided.")
            cond_vec = cond_vec + self.instruction_proj(instruction)

        cond_tokens = self._pad_tokens(x)
        cond_tokens = self.cond_proj(cond_tokens)
        cond_vec = cond_vec + cond_tokens.mean(dim=1)

        for block in self.blocks:
            xt_tokens = block(xt_tokens, cond_tokens, cond_vec)

        xt_tokens = self.norm_out(xt_tokens)
        out_tokens = self.head(xt_tokens)

        h = self.image_width // self.patch_size
        w = h
        out = out_tokens.view(batch_size, h, w, self.out_chans, self.patch_size, self.patch_size)
        out = out.permute(0, 3, 1, 4, 2, 5).reshape(
            batch_size,
            self.out_chans,
            self.image_width,
            self.image_width,
        )
        return out
