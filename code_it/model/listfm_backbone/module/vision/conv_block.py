from enum import Enum

import torch
from torch import Tensor, nn
from torch.nn import functional


class BlockType(str, Enum):
    BLOCK1 = "block1"
    BLOCK2 = "block2"
    BLOCK3 = "block3"

    @classmethod
    def from_string(cls, value: str) -> "BlockType":
        try:
            return cls(value)
        except ValueError as err:
            raise ValueError(f"Invalid BlockType value: {value}. Must be one of {list(cls)} : {err}") from err


class ConvAttentionBlock1(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        head_dim: int = 32,
    ):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

        self.num_heads = out_chans // head_dim
        self.scale = head_dim**-0.5
        self.qkv = nn.Conv2d(out_chans, out_chans * 3, kernel_size=1)
        self.proj = nn.Conv2d(out_chans, out_chans, kernel_size=1)
        self.attention_norm = nn.GroupNorm(4, out_chans)

        self.layer3 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

    def shifted_window_attention(
        self,
        x: torch.Tensor,
        window_size: int = 8,
    ) -> torch.Tensor:
        _B, _C, H, W = x.shape

        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size

        x = functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        shift_size = window_size // 2
        x_shifted = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))

        x1 = self._apply_window_attention(x, window_size)
        x2 = self._apply_window_attention(x_shifted, window_size)

        x2 = torch.roll(x2, shifts=(shift_size, shift_size), dims=(2, 3))

        x = (x1 + x2) * 0.5
        x = x[:, :, :H, :W]
        return x

    def _apply_window_attention(
        self,
        x: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        B, C, H, W = x.shape

        x = x.unfold(2, window_size, window_size).unfold(3, window_size, window_size)
        x = x.contiguous().view(B, C, -1, window_size, window_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        num_windows = x.shape[1]
        x = x.view(-1, C, window_size, window_size)

        qkv = self.qkv(x).reshape(-1, 3, self.num_heads, C // self.num_heads, window_size * window_size)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(-1, C, window_size, window_size)

        x = self.proj(x)

        x = x.view(B, num_windows, C, window_size, window_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        x = x.view(B, C, H, W)
        return x

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:

        x = self.layer1(x)

        x = self.layer2(x)

        residual = x
        x = self.attention_norm(x)
        x = self.shifted_window_attention(x)
        x = residual + x

        x = self.layer3(x)

        return x


class ConvAttentionBlock2(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
    ) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_chans, out_chans // 4, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_chans // 4, out_chans, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        x = self.layer1(x)
        se_weight = self.se_block(x)
        x = x * se_weight
        x = self.layer2(x)
        return x


class ConvAttentionBlock3(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
    ) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def get_time_conv_attention_block(
    block_type: BlockType,
    in_chans: int,
    out_chans: int,
) -> nn.Module:
    if block_type == BlockType.BLOCK1:
        return ConvAttentionBlock1(
            in_chans=in_chans,
            out_chans=out_chans,
        )
    elif block_type == BlockType.BLOCK2:
        return ConvAttentionBlock2(
            in_chans=in_chans,
            out_chans=out_chans,
        )
    elif block_type == BlockType.BLOCK3:
        return ConvAttentionBlock3(
            in_chans=in_chans,
            out_chans=out_chans,
        )
    else:
        raise ValueError(f"Unknown block type: {block_type}")
