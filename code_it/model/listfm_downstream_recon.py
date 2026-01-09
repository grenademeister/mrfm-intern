from pathlib import Path

import torch
from torch import Tensor

from model.listfm_backbone import LISTFMConfig, LISTFoundationModelBackbone
from model.listfm_backbone.module import (
    Bottleneck,
    SimpleTokenizer,
    TextEncoder,
    VisionDecoder,
    VisionEncoder,
)
from model.listfm_backbone.utils import (
    logger,
    validate_tensor_channels,
    validate_tensor_dimensions,
    validate_tensors,
)


class LISTFoundationModelDownstreamRecon(LISTFoundationModelBackbone):
    listfmconfig: LISTFMConfig  # predefined
    vision_encoder: VisionEncoder  # predefined
    text_encoder: TextEncoder  # predefined
    tokenizer: SimpleTokenizer  # predefined
    bottleneck: Bottleneck  # predefined
    vision_decoder: VisionDecoder

    def __init__(
        self,
        listfmconfig: LISTFMConfig,
        use_vision_decoder: bool,
    ) -> None:
        super().__init__(
            listfmconfig=listfmconfig,
            use_vision_decoder=use_vision_decoder,
        )

    def forward(
        self,
        img: Tensor,
        text: Tensor,
        grad_encoder: bool = True,
        use_bottleneck: bool = True,
        instruction: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        _ = instruction
        validate_tensors([img, text])
        validate_tensor_dimensions([img], 4)
        validate_tensor_channels(img, self.listfmconfig.img_in_chan)

        img = self._preprocess_image(img=img)
        if not grad_encoder:
            with torch.no_grad():
                (
                    _img_feature,
                    img_full_feature,
                    stack_feature,
                ) = self.vision_encoder(
                    x=img,
                )
        else:
            (
                _img_feature,
                img_full_feature,
                stack_feature,
            ) = self.vision_encoder(
                x=img,
            )

        if use_bottleneck:
            with torch.no_grad():
                (
                    _text_features,
                    _text_full_feature,
                ) = self.text_encoder(
                    text=text,
                )
            (
                img_full_feature,
                _text_full_feature,
            ) = self.bottleneck(
                vision_feature=img_full_feature,
                text_feature=_text_full_feature,
            )

        img_decode = self.vision_decoder(
            x=img_full_feature,
            stack_feat=stack_feature,
        )

        return img_decode


def load_from_ckpt(
    ckpt_path: Path,
    from_scratch: bool = False,
    use_vision_decoder: bool = True,
    use_vision_decoder_weights: bool = True,
) -> LISTFoundationModelDownstreamRecon:
    # Validation
    if use_vision_decoder and not use_vision_decoder_weights:
        logger.warning(
            "use_vision_decoder_weights is set to False while use_vision_decoder is True. The vision decoder will be randomly initialized."
        )
    if not use_vision_decoder and use_vision_decoder_weights:
        raise ValueError("use_vision_decoder_weights cannot be True if use_vision_decoder is False")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist.")

    # Load checkpoint
    longitudinal_checkpoint_data = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    # Validate checkpoint
    if not (("model_state_dict" in longitudinal_checkpoint_data) and ("model_config" in longitudinal_checkpoint_data)):
        raise KeyError("Invalid Checkpoint")

    logger.success("Checkpoint loaded successfully.")

    # Load model config
    listfmconfig = LISTFMConfig(**longitudinal_checkpoint_data["model_config"])
    listfmconfig.text_enc_pretrained = None

    # Load model state dict
    model_state_dict = {}
    for key in longitudinal_checkpoint_data["model_state_dict"]:
        new_key = key
        if key.startswith("module."):
            new_key = key[7:]

        if new_key.split(".")[0] in [
            "vision_encoder",
            "text_encoder",
            "bottleneck",
        ]:
            model_state_dict[new_key] = longitudinal_checkpoint_data["model_state_dict"][key]

        if use_vision_decoder_weights and new_key.split(".")[0] == "vision_decoder":
            model_state_dict[new_key] = longitudinal_checkpoint_data["model_state_dict"][key]

    # Initialize model
    model = LISTFoundationModelDownstreamRecon(
        listfmconfig,
        use_vision_decoder=use_vision_decoder,
    )

    # Load state dict
    if from_scratch:
        logger.warning("Loading model from scratch. All weights will be randomly initialized.")
    else:
        model.load_state_dict(model_state_dict, strict=False)
    model.qc()

    logger.success("Model state dict loaded successfully.")

    return model
