import torch
from torch import nn
from torch.nn import functional as F

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


class FiLM(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        out_chans: int,
    ) -> None:
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, out_chans * 2)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self.to_scale_shift(cond).chunk(2, dim=-1)
        gamma = gamma.view(cond.shape[0], x.shape[1], 1, 1)
        beta = beta.view(cond.shape[0], x.shape[1], 1, 1)
        return x * (1 + gamma) + beta


class InstructionAdapter(nn.Module):
    def __init__(
        self,
        chans: int,
        instruction_dim: int,
        bottleneck_ratio: float = 0.25,
        min_chans: int = 8,
    ) -> None:
        super().__init__()
        hidden_chans = max(min_chans, int(chans * bottleneck_ratio))
        self.down = nn.Conv2d(chans, hidden_chans, kernel_size=1, padding=0)
        self.act = nn.SiLU(inplace=True)
        self.up = nn.Conv2d(hidden_chans, chans, kernel_size=1, padding=0)
        self.gate = nn.Linear(instruction_dim, chans)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        x: torch.Tensor,
        instruction: torch.Tensor,
    ) -> torch.Tensor:
        if instruction is None:
            return x
        gate = torch.tanh(self.gate(instruction))
        gate = gate.view(instruction.shape[0], x.shape[1], 1, 1)
        return x + self.up(self.act(self.down(x))) * gate


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
    time_film_layers: nn.ModuleList
    instruction_adapters: nn.ModuleList

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
        self.instruction_dim = instruction_dim

        self.instruction_pool = None
        if instruction_dim is not None:
            self.instruction_pool = nn.Sequential(
                nn.Linear(instruction_dim, instruction_dim),
                nn.SiLU(inplace=True),
                nn.Linear(instruction_dim, 1),
            )

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

        self.stage_out_chans = []
        ch = decoder_feature_chans * (2 ** (num_pool_layers - 1))
        for _ in range(num_pool_layers - 1):
            self.stage_out_chans.append(ch // 2)
            ch //= 2
        self.stage_out_chans.append(ch)

        self.time_film_layers = nn.ModuleList()
        for out_ch in self.stage_out_chans:
            self.time_film_layers.append(FiLM(self.init_chan, out_ch))

        self.instruction_adapters = nn.ModuleList()
        if instruction_dim is not None:
            for out_ch in self.stage_out_chans:
                self.instruction_adapters.append(InstructionAdapter(out_ch, instruction_dim))

        # Final convolution layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_feature_chans, decoder_feature_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, decoder_feature_chans),
            nn.SiLU(inplace=True),
            nn.Conv2d(decoder_feature_chans, out_chans, kernel_size=1, padding=0),
        )

    def _prepare_instruction(
        self,
        instruction: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if instruction is None:
            return None
        if instruction.dim() == 2:
            raise ValueError("instruction must be a 3D (B, N, D) tensor.")
        if instruction.dim() != 3:
            raise ValueError("instruction must be a 3D (B, N, D) tensor.")
        if self.instruction_pool is None:
            raise RuntimeError("instruction_dim must be set when instruction is provided.")
        weights = self.instruction_pool(instruction).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        return torch.sum(instruction * weights.unsqueeze(-1), dim=1)

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

    def _reshape_encoder_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        output = x.permute(0, 2, 1)
        output = output[:, : self.encoder_init_chan, : self.embed_w**2]
        output = output.reshape(batch_size, self.encoder_init_chan, self.embed_w, self.embed_w)
        output = self.feat_proj(output)
        return output

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

        batch_size = x.shape[0]
        instruction = self._prepare_instruction(instruction)
        if instruction is not None:
            if len(self.instruction_adapters) != len(self.up_sample_layers):
                raise RuntimeError("Instruction adapters not configured for conditioning.")
            if self.instruction_dim is None:
                raise RuntimeError("instruction_dim must be set when instruction is provided.")

        output = self._reshape_encoder_features(x)  # (B, C_dec0, Ew, Ew)

        flow_xt = F.interpolate(flow_xt, size=(self.embed_w, self.embed_w), mode="bilinear", align_corners=False)  # (B, C_in, Ew, Ew)
        output = output + self.xt_proj(flow_xt)  # (B, C_dec0, Ew, Ew)

        flow_t = self._prepare_flow_t(flow_t, batch_size)
        time_emb = self.time_mlp(flow_t)  # (B, C_dec0)

        for i, layer in enumerate(self.up_sample_layers):
            downsampled_output = stack_feat.pop()
            layer_size = downsampled_output.shape[-2:]
            output = F.interpolate(output, size=layer_size, mode="bilinear", align_corners=False)  # (B, C_dec_i, S_i, S_i)
            flow_xt_scaled = F.interpolate(flow_xt, size=layer_size, mode="bilinear", align_corners=False)  # (B, C_in, S_i, S_i)
            downsampled_output = self.skip_proj[i](downsampled_output)  # (B, C_dec_i, S_i, S_i)
            xt_feat = self.xt_proj_layers[i](flow_xt_scaled)  # (B, C_dec_i, S_i, S_i)
            output = torch.cat([output, downsampled_output, xt_feat], dim=1)  # (B, 3*C_dec_i, S_i, S_i)
            output = layer(output)  # (B, C_dec_{i+1}, S_i, S_i)
            output = self.time_film_layers[i](output, time_emb)  # (B, C_dec_{i+1}, S_i, S_i)
            if instruction is not None:
                output = self.instruction_adapters[i](output, instruction)  # (B, C_dec_{i+1}, S_i, S_i)

        output = self.final_conv(output)  # (B, C_out, W, W)
        return output
