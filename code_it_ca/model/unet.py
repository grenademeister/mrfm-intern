import torch
from torch import nn
from torch.nn import functional as f


def create_down_sample_layers(
    in_chans: int,
    chans: int,
    num_pool_layers: int,
) -> nn.ModuleList:
    layers = nn.ModuleList([ConvBlock(in_chans, chans)])
    ch = chans
    for _ in range(num_pool_layers - 1):
        layers.append(ConvBlock(ch, ch * 2))
        ch *= 2
    return layers


def create_up_sample_layers(
    chans: int,
    num_pool_layers: int,
) -> nn.ModuleList:
    layers = nn.ModuleList()
    ch = chans * (2 ** (num_pool_layers - 1))
    for _ in range(num_pool_layers - 1):
        layers.append(ConvBlock(ch * 2, ch // 2))
        ch //= 2
    layers.append(ConvBlock(ch * 2, ch))
    return layers


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(x)


class Unet(nn.Module):
    def __init__(
        self,
        in_chans: int = 2,
        out_chans: int = 2,
        chans: int = 64,
        num_pool_layers: int = 5,
    ) -> None:
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_pool_layers = num_pool_layers

        # Down-sampling layers
        self.down_sample_layers = create_down_sample_layers(
            in_chans,
            chans,
            num_pool_layers,
        )

        # # Bottleneck layer
        self.bottleneck_conv = ConvBlock(chans * (2 ** (num_pool_layers - 1)), chans * (2 ** (num_pool_layers - 1)))

        # self.transformer = Transformer(
        #     width=chans * (2 ** (num_pool_layers - 1)),
        #     layers=12,
        #     heads=8,
        #     attn_mask=None,
        # )

        # Up-sampling layers
        self.up_sample_layers = create_up_sample_layers(
            chans,
            num_pool_layers,
        )

        # Final convolution layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(chans, chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, chans),
            nn.SiLU(inplace=True),
            nn.Conv2d(chans, out_chans, kernel_size=1, padding=0),
        )

    def forward(
        self,
        img: torch.Tensor,
        text: torch.Tensor,
        use_bottleneck: bool = True,
        grad_encoder: bool = True,
    ) -> torch.Tensor:
        stack = []
        output = img
        # Down-sampling
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = f.max_pool2d(output, kernel_size=2)

        # Bottleneck
        B, C, W, H = output.shape

        output = output.reshape(B, C, -1)
        output = output.permute(0, 2, 1)
        output = self.transformer(output)
        output = output.permute(0, 2, 1)
        output = output.reshape(B, C, W, H)

        # Up-sampling
        for layer in self.up_sample_layers:
            downsampled_output = stack.pop()
            layer_size = downsampled_output.shape[-2:]
            output = f.interpolate(output, size=layer_size, mode="bilinear", align_corners=False)
            output = torch.cat([output, downsampled_output], dim=1)
            output = layer(output)

        output = self.final_conv(output)
        return output
