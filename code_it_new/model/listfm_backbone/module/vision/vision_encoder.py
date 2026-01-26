import torch
from torch import nn

from ..transformer import LayerNorm, Transformer
from .conv_block import BlockType, get_time_conv_attention_block


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        block_type: BlockType,
    ) -> None:
        super().__init__()

        self.layers = get_time_conv_attention_block(
            block_type=block_type,
            in_chans=in_chans,
            out_chans=out_chans,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(x)


def create_down_sample_layers(
    in_chans: int,
    chans: int,
    num_pool_layers: int,
    block_type: BlockType,
) -> nn.ModuleList:
    layers = nn.ModuleList([ConvBlock(in_chans, chans, block_type)])
    ch = chans
    for _ in range(num_pool_layers - 1):
        layers.append(ConvBlock(ch, ch * 2, block_type))
        ch *= 2
    return layers


class VisionEncoder(nn.Module):
    in_chans: int
    num_pool_layers: int
    down_sample_layers: nn.ModuleList
    transformer: Transformer

    def __init__(
        self,
        in_chans: int = 2,
        feature_chans: int = 64,
        num_pool_layers: int = 5,
        transformer_layers: int = 12,
        transformer_n_head: int = 8,
        image_width: int = 512,
        output_dim: int = 512,
        block_type: BlockType = BlockType.BLOCK3,
    ) -> None:
        super().__init__()

        self.in_chans = in_chans
        self.num_pool_layers = num_pool_layers

        self.embed_dim = feature_chans * (2 ** (num_pool_layers - 1))
        self.embed_w = image_width // (2 ** (num_pool_layers))

        self.down_sample_layers = create_down_sample_layers(
            in_chans=in_chans,
            chans=feature_chans,
            num_pool_layers=num_pool_layers,
            block_type=block_type,
        )

        self.transformer = Transformer(
            width=self.embed_dim,
            layers=transformer_layers,
            heads=transformer_n_head,
            attn_mask=None,
        )

        scale = self.embed_dim**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(self.embed_dim))
        self.positional_embedding = nn.Parameter(scale * torch.randn(1, self.embed_w**2 + 1, self.embed_dim))
        self.ln_pre = LayerNorm(self.embed_dim)
        self.ln_post = LayerNorm(self.embed_dim)
        self.proj = nn.Parameter(scale * torch.randn(self.embed_dim, output_dim))

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x_dim = x.dim()
        if x_dim != 4:
            raise NotImplementedError("x dim should be 4")

        stack: list[torch.Tensor] = []
        output = x

        # Down-sampling
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = torch.nn.functional.max_pool2d(output, kernel_size=2)

        # Transformer
        B, C, _, _ = output.shape

        output = output.reshape(B, C, -1)
        output = output.permute(0, 2, 1)

        class_token = self.class_embedding.to(output.dtype).expand(B, 1, -1)
        output = torch.cat([class_token, output], dim=1)

        output = output + self.positional_embedding[:, : output.shape[1], :].to(output.device)
        output = self.ln_pre(output)

        output = self.transformer(output)
        full_feature = output

        output = self.ln_post(output[:, 0, :])
        output = output @ self.proj

        return output, full_feature, stack
