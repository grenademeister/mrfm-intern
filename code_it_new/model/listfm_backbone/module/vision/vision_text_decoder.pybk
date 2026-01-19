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


def create_up_sample_layers_with_xt(
    chans: int,
    num_pool_layers: int,
    block_type: BlockType,
) -> nn.ModuleList:
    layers = nn.ModuleList()
    ch = chans * (2 ** (num_pool_layers - 1))
    for _ in range(num_pool_layers - 1):
        layers.append(ConvBlock(ch * 3, ch // 2, block_type))
        ch //= 2
    layers.append(ConvBlock(ch * 3, ch, block_type))
    return layers


class VisionTextDecoder(nn.Module):
    out_chans: int
    num_pool_layers: int
    up_sample_layers: nn.ModuleList
    final_conv: nn.Sequential
    time_mlp: nn.Sequential
    xt_proj: nn.Sequential
    skip_proj: nn.ModuleList
    feat_proj: nn.Module
    xt_proj_layers: nn.ModuleList

    def __init__(
        self,
        out_chans: int = 2,
        feature_chans: int = 64,
        decoder_feature_chans: int | None = None,
        num_pool_layers: int = 5,
        image_width: int = 512,
        block_type: BlockType = BlockType.BLOCK2,
        instruction_dim: int | None = None,
        input_chans: int | None = None,
    ) -> None:
        super().__init__()

        self.out_chans = out_chans
        self.num_pool_layers = num_pool_layers
        self.encoder_init_chan = feature_chans * (2 ** (num_pool_layers - 1))
        if decoder_feature_chans is None:
            decoder_feature_chans = feature_chans
        self.decoder_feature_chans = decoder_feature_chans
        self.init_chan = decoder_feature_chans * (2 ** (num_pool_layers - 1))
        self.embed_w = image_width // (2 ** (num_pool_layers))
        self.input_chans = input_chans if input_chans is not None else out_chans

        # Up-sampling layers
        self.up_sample_layers = create_up_sample_layers_with_xt(
            chans=decoder_feature_chans,
            num_pool_layers=num_pool_layers,
            block_type=block_type,
        )

        self.skip_proj = nn.ModuleList()
        for i in range(num_pool_layers):
            enc_ch = feature_chans * (2 ** (num_pool_layers - 1 - i))
            dec_ch = decoder_feature_chans * (2 ** (num_pool_layers - 1 - i))
            self.skip_proj.append(
                nn.Sequential(
                    nn.Conv2d(enc_ch, dec_ch, kernel_size=1, padding=0),
                    nn.GroupNorm(4, dec_ch),
                    nn.SiLU(inplace=True),
                )
            )

        if self.encoder_init_chan == self.init_chan:
            self.feat_proj = nn.Identity()
        else:
            self.feat_proj = nn.Sequential(
                nn.Conv2d(self.encoder_init_chan, self.init_chan, kernel_size=1, padding=0),
                nn.GroupNorm(4, self.init_chan),
                nn.SiLU(inplace=True),
            )

        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.init_chan),
            nn.SiLU(inplace=True),
            nn.Linear(self.init_chan, self.init_chan),
        )

        self.xt_proj = nn.Sequential(
            nn.Conv2d(self.input_chans, self.init_chan, kernel_size=1, padding=0),
            nn.GroupNorm(4, self.init_chan),
            nn.SiLU(inplace=True),
        )

        self.xt_proj_layers = nn.ModuleList()
        ch = decoder_feature_chans * (2 ** (num_pool_layers - 1))
        for _ in range(num_pool_layers - 1):
            self.xt_proj_layers.append(
                nn.Sequential(
                    nn.Conv2d(self.input_chans, ch, kernel_size=1, padding=0),
                    nn.GroupNorm(4, ch),
                    nn.SiLU(inplace=True),
                )
            )
            ch //= 2
        self.xt_proj_layers.append(
            nn.Sequential(
                nn.Conv2d(self.input_chans, ch, kernel_size=1, padding=0),
                nn.GroupNorm(4, ch),
                nn.SiLU(inplace=True),
            )
        )

        self.film_layers = nn.ModuleList()
        if instruction_dim is not None:
            block_out_chans = []
            ch = decoder_feature_chans * (2 ** (num_pool_layers - 1))
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
            nn.Conv2d(decoder_feature_chans, decoder_feature_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, decoder_feature_chans),
            nn.SiLU(inplace=True),
            nn.Conv2d(decoder_feature_chans, out_chans, kernel_size=1, padding=0),
        )

    def forward(
        self,
        x: torch.Tensor,
        stack_feat: list[torch.Tensor],
        instruction: torch.Tensor = None,
        flow_xt: torch.Tensor | None = None,
        flow_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_dim = x.dim()
        if x_dim != 3:
            raise NotImplementedError("x dim should be 3")
        if flow_xt is None or flow_t is None:
            raise ValueError("flow_xt and flow_t must be provided for rectified-flow decoding.")

        output = x

        H, W = self.embed_w, self.embed_w
        C = self.init_chan
        B = x.shape[0]

        output = output.permute(0, 2, 1)
        output = output[:, : self.encoder_init_chan, : self.embed_w**2]
        output = output.reshape(B, self.encoder_init_chan, H, W)
        output = self.feat_proj(output)

        flow_xt = torch.nn.functional.interpolate(flow_xt, size=(H, W), mode="bilinear", align_corners=False)
        output = output + self.xt_proj(flow_xt)

        if flow_t.dim() == 4:
            flow_t = flow_t.view(B, -1)
        if flow_t.dim() != 2 or flow_t.shape[1] != 1:
            raise ValueError("flow_t must be a (B, 1) or (B, 1, 1, 1) tensor.")
        time_emb = self.time_mlp(flow_t).view(B, C, 1, 1)
        output = output + time_emb

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
            flow_xt_scaled = torch.nn.functional.interpolate(flow_xt, size=layer_size, mode="bilinear", align_corners=False)
            downsampled_output = self.skip_proj[i](downsampled_output)
            xt_feat = self.xt_proj_layers[i](flow_xt_scaled)
            output = torch.cat([output, downsampled_output, xt_feat], dim=1)
            output = layer(output)
            if instruction is not None:
                film = self.film_layers[i](instruction)
                gamma, beta = film.chunk(2, dim=-1)
                gamma = gamma.view(B, output.shape[1], 1, 1)
                beta = beta.view(B, output.shape[1], 1, 1)
                output = output * (1 + gamma) + beta

        output = self.final_conv(output)
        return output
