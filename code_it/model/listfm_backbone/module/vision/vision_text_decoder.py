import torch
from torch import nn

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


def create_up_sample_layers(
    chans: int,
    num_pool_layers: int,
    block_type: BlockType,
) -> nn.ModuleList:
    layers = nn.ModuleList()
    ch = chans * (2 ** (num_pool_layers - 1))
    for _ in range(num_pool_layers - 1):
        layers.append(ConvBlock(ch * 2, ch // 2, block_type))
        ch //= 2
    layers.append(ConvBlock(ch * 2, ch, block_type))
    return layers


class VisionTextDecoder(nn.Module):
    out_chans: int
    num_pool_layers: int
    up_sample_layers: nn.ModuleList
    final_conv: nn.Sequential

    def __init__(
        self,
        out_chans: int = 2,
        feature_chans: int = 64,
        num_pool_layers: int = 5,
        image_width: int = 512,
        block_type: BlockType = BlockType.BLOCK2,
        instruction_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.out_chans = out_chans
        self.num_pool_layers = num_pool_layers
        self.init_chan = feature_chans * (2 ** (num_pool_layers - 1))
        self.embed_w = image_width // (2 ** (num_pool_layers))

        # Up-sampling layers
        self.up_sample_layers = create_up_sample_layers(
            chans=feature_chans,
            num_pool_layers=num_pool_layers,
            block_type=block_type,
        )

        self.film_layers = nn.ModuleList()
        if instruction_dim is not None:
            block_out_chans = []
            ch = feature_chans * (2 ** (num_pool_layers - 1))
            for _ in range(num_pool_layers - 1):
                block_out_chans.append(ch // 2)
                ch //= 2
            block_out_chans.append(ch)
            for out_ch in block_out_chans:
                film = nn.Linear(instruction_dim, out_ch * 2)
                nn.init.zeros_(film.weight)
                nn.init.zeros_(film.bias)
                self.film_layers.append(film)

        # Final convolution layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(feature_chans, feature_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, feature_chans),
            nn.SiLU(inplace=True),
            nn.Conv2d(feature_chans, out_chans, kernel_size=1, padding=0),
        )

    def forward(
        self,
        x: torch.Tensor,
        stack_feat: list[torch.Tensor],
        instruction: torch.Tensor = None,
    ) -> torch.Tensor:
        x_dim = x.dim()
        if x_dim != 3:
            raise NotImplementedError("x dim should be 3")

        output = x

        H, W = self.embed_w, self.embed_w
        C = self.init_chan
        B = x.shape[0]

        output = output.permute(0, 2, 1)
        output = output[:, :C, : self.embed_w**2]
        output = output.reshape(B, C, H, W)

        if instruction is not None:
            if instruction.dim() == 3:
                instruction = instruction.mean(dim=1)
            if instruction.dim() != 2:
                raise ValueError("instruction must be 2D or 3D tensor.")
            if len(self.film_layers) != len(self.up_sample_layers):
                raise RuntimeError("FiLM layers not configured for instruction conditioning.")

        for i, layer in enumerate(self.up_sample_layers):
            downsampled_output = stack_feat.pop()
            layer_size = downsampled_output.shape[-2:]
            output = torch.nn.functional.interpolate(output, size=layer_size, mode="bilinear", align_corners=False)
            output = torch.cat([output, downsampled_output], dim=1)
            output = layer(output)
            if instruction is not None:
                film = self.film_layers[i](instruction)
                gamma, beta = film.chunk(2, dim=-1)
                gamma = gamma.view(B, output.shape[1], 1, 1)
                beta = beta.view(B, output.shape[1], 1, 1)
                output = output * (1 + gamma) + beta

        output = self.final_conv(output)
        return output
