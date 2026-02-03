import torch
from torch import nn
from torch.nn import functional as F


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


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.res1 = ResidualBlock(in_ch + skip_ch, out_ch)
        self.res2 = ResidualBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x)
        x = self.res2(x)
        return x


class UpSampleBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.res = ResidualBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.res(x)


class VisionTextDecoder(nn.Module):
    out_chans: int

    def __init__(
        self,
        out_chans: int = 2,
        feature_chans: int = 64,
        decoder_feature_chans: int | None = None,
        num_pool_layers: int = 5,
        image_width: int = 512,
        instruction_dim: int | None = None,
        input_chans: int | None = None,
    ) -> None:
        super().__init__()

        self.out_chans = out_chans
        base_channels = decoder_feature_chans if decoder_feature_chans is not None else feature_chans
        if base_channels < 64:
            base_channels = 64
        self.base_channels = base_channels
        self.input_chans = input_chans if input_chans is not None else out_chans
        self.instruction_dim = instruction_dim

        self.instruction_pool = None
        if instruction_dim is not None:
            self.instruction_pool = nn.Sequential(
                nn.Linear(instruction_dim, instruction_dim),
                nn.SiLU(inplace=True),
                nn.Linear(instruction_dim, 1),
            )
        self.stem = nn.LazyConv2d(base_channels, kernel_size=3, padding=1)
        self.enc1 = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels),
        )
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2),
        )
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 4),
        )
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)

        up_channels = [
            max(32, base_channels // 2),
            max(32, base_channels // 4),
            max(32, base_channels // 8),
            max(32, base_channels // 16),
        ]
        up_blocks = []
        in_ch = base_channels
        for out_ch in up_channels:
            up_blocks.append(UpSampleBlock(in_ch, out_ch))
            in_ch = out_ch
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = nn.Conv2d(in_ch, out_chans, kernel_size=1)

        self.flow_stem = nn.Sequential(
            nn.Conv2d(self.input_chans, base_channels, kernel_size=3, padding=1),
            ResidualBlock(base_channels, base_channels),
        )
        self.flow_down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            ResidualBlock(base_channels * 2, base_channels * 2),
        )
        self.flow_down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            ResidualBlock(base_channels * 4, base_channels * 4),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(1, base_channels),
            nn.SiLU(inplace=True),
            nn.Linear(base_channels, base_channels),
        )

        self.time_film_1 = FiLM(base_channels, base_channels)
        self.time_film_2 = FiLM(base_channels, base_channels * 2)
        self.time_film_3 = FiLM(base_channels, base_channels * 4)
        self.time_film_4 = FiLM(base_channels, base_channels * 2)
        self.time_film_5 = FiLM(base_channels, base_channels)

        self.instruction_adapters = nn.ModuleList()
        if instruction_dim is not None:
            self.instruction_adapters.extend(
                [
                    InstructionAdapter(base_channels, instruction_dim),
                    InstructionAdapter(base_channels * 2, instruction_dim),
                    InstructionAdapter(base_channels * 4, instruction_dim),
                    InstructionAdapter(base_channels * 2, instruction_dim),
                    InstructionAdapter(base_channels, instruction_dim),
                ]
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

    def forward(
        self,
        x: torch.Tensor,
        stack_feat: list[torch.Tensor],
        instruction: torch.Tensor = None,
        flow_xt: torch.Tensor | None = None,
        flow_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_dim = x.dim()
        if x_dim != 4:
            raise NotImplementedError("x dim should be 4")
        if flow_xt is None or flow_t is None:
            raise ValueError("flow_xt and flow_t must be provided for rectified-flow decoding.")

        batch_size = x.shape[0]
        instruction = self._prepare_instruction(instruction)
        if instruction is not None:
            if self.instruction_dim is None:
                raise RuntimeError("instruction_dim must be set when instruction is provided.")

        target_size = flow_xt.shape[-2:]
        output = x.permute(0, 3, 1, 2).contiguous()

        flow_feat1 = self.flow_stem(flow_xt)
        flow_feat2 = self.flow_down1(flow_feat1)
        flow_feat3 = self.flow_down2(flow_feat2)

        output = self.stem(output)
        output = output + F.interpolate(flow_feat1, size=output.shape[-2:], mode="bilinear", align_corners=False)

        flow_t = self._prepare_flow_t(flow_t, batch_size)
        time_emb = self.time_mlp(flow_t)

        output = self.enc1(output)
        output = self.time_film_1(output, time_emb)
        output = output + F.interpolate(flow_feat1, size=output.shape[-2:], mode="bilinear", align_corners=False)
        if instruction is not None:
            output = self.instruction_adapters[0](output, instruction)
        skip1 = output

        output = self.down1(output)
        output = self.enc2(output)
        output = self.time_film_2(output, time_emb)
        output = output + F.interpolate(flow_feat2, size=output.shape[-2:], mode="bilinear", align_corners=False)
        if instruction is not None:
            output = self.instruction_adapters[1](output, instruction)
        skip2 = output

        output = self.down2(output)
        output = self.bottleneck(output)
        output = self.time_film_3(output, time_emb)
        output = output + F.interpolate(flow_feat3, size=output.shape[-2:], mode="bilinear", align_corners=False)
        if instruction is not None:
            output = self.instruction_adapters[2](output, instruction)

        output = self.up2(output, skip2)
        output = self.time_film_4(output, time_emb)
        output = output + F.interpolate(flow_feat2, size=output.shape[-2:], mode="bilinear", align_corners=False)
        if instruction is not None:
            output = self.instruction_adapters[3](output, instruction)

        output = self.up1(output, skip1)
        output = self.time_film_5(output, time_emb)
        output = output + F.interpolate(flow_feat1, size=output.shape[-2:], mode="bilinear", align_corners=False)
        if instruction is not None:
            output = self.instruction_adapters[4](output, instruction)

        for block in self.up_blocks:
            output = block(output)
        output = self.out(output)
        if output.shape[-2:] != target_size:
            output = F.interpolate(output, size=target_size, mode="bilinear", align_corners=False)
        return output
