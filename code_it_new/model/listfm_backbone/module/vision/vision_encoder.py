import math

import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel


def get_patch_tokens(
    last_hidden_state: torch.Tensor,
    num_register_tokens: int = 0,
) -> torch.Tensor:
    _, tokens, _ = last_hidden_state.shape
    tokens_minus_special = tokens - (1 + num_register_tokens)
    grid_minus = int(math.sqrt(tokens_minus_special))
    if grid_minus * grid_minus == tokens_minus_special:
        start = 1 + num_register_tokens
        return last_hidden_state[:, start:, :]

    tokens_minus_cls = tokens - 1
    grid_minus_cls = int(math.sqrt(tokens_minus_cls))
    if grid_minus_cls * grid_minus_cls == tokens_minus_cls:
        return last_hidden_state[:, 1:, :]

    grid_full = int(math.sqrt(tokens))
    if grid_full * grid_full == tokens:
        return last_hidden_state

    raise ValueError(f"Unexpected token count: {tokens}")


class VisionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        finetune_encoder: bool = True,
        input_size: int | None = None,
        input_is_normalized: bool = True,
    ) -> None:
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = finetune_encoder
        if not finetune_encoder:
            self.encoder.eval()

        self.input_size = input_size
        self.input_is_normalized = input_is_normalized

    def _prepare_inputs(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() != 4:
            raise NotImplementedError("x dim should be 4")

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 2:
            x = torch.cat([x, x[:, :1]], dim=1)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected 1, 2, or 3 channels, got {x.shape[1]}")

        kwargs = {}
        kwargs["do_resize"] = False
        if self.input_is_normalized:
            kwargs["do_rescale"] = False
            kwargs["do_normalize"] = False
        inputs = self.processor(
            images=x,
            return_tensors="pt",
            input_data_format="channels_first",
            **kwargs,
        )
        pixel_values = inputs["pixel_values"].to(device=x.device, dtype=x.dtype)
        return pixel_values

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        pixel_values = self._prepare_inputs(x)
        outputs = self.encoder(pixel_values)
        num_register_tokens = getattr(self.encoder.config, "num_register_tokens", 0)
        patch_tokens = get_patch_tokens(outputs.last_hidden_state, num_register_tokens)
        batch, tokens, dim = patch_tokens.shape
        grid = int(math.sqrt(tokens))
        if grid * grid != tokens:
            raise ValueError(f"Patch tokens do not form a square grid: {tokens}")
        return patch_tokens.reshape(batch, grid, grid, dim)


if __name__ == "__main__":
    model = VisionEncoder()
    x = torch.randn(2, 2, 512, 512)
    out = model(x)
    print(f"out shape: {out.shape}")
