import torch
from torch import nn

from ..transformer import LayerNorm, Transformer


class Bottleneck(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        context_length: int,
    ) -> None:
        super().__init__()
        self.width = width
        self.layers = layers

        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
        )

        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.ln_final = LayerNorm(width)

    def forward(
        self,
        vision_feature: torch.Tensor,
        text_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        vision_feature, text_feature = self.width_pad(
            vision_feature=vision_feature,
            text_feature=text_feature,
        )

        x = torch.cat((vision_feature, text_feature), dim=1)
        x = x + self.positional_embedding.type(torch.float32)
        x = self.transformer(x)
        x = self.ln_final(x).type(torch.float32)
        vision_output = x[:, : vision_feature.shape[1], :]
        text_output = x[:, vision_feature.shape[1] :, :]
        return vision_output, text_output

    def width_pad(
        self,
        vision_feature: torch.Tensor,
        text_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vision_feature_chan = vision_feature.shape[2]
        text_feature_chan = text_feature.shape[2]

        if vision_feature_chan < self.width:
            pad_vision = torch.zeros(
                size=(vision_feature.shape[0], vision_feature.shape[1], self.width - vision_feature_chan),
                device=vision_feature.device,
                dtype=vision_feature.dtype,
            )
            vision_feature = torch.cat((vision_feature, pad_vision), dim=2)
        else:
            vision_feature = vision_feature[:, :, : self.width]

        if text_feature_chan < self.width:
            pad_text = torch.zeros(
                size=(text_feature.shape[0], text_feature.shape[1], self.width - text_feature_chan),
                device=text_feature.device,
                dtype=text_feature.dtype,
            )
            text_feature = torch.cat((text_feature, pad_text), dim=2)
        else:
            text_feature = text_feature[:, :, : self.width]

        return vision_feature, text_feature
