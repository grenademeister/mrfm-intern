from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from model.listfm_backbone import LISTFMConfig, LISTFoundationModelBackbone
from model.listfm_backbone.module import (
    Bottleneck,
    QwenInstructionEncoder,
    SimpleTokenizer,
    TextEncoder,
    VisionTextDecoder,
    VisionEncoder,
)
from model.listfm_backbone.utils import (
    logger,
    validate_tensor_channels,
    validate_tensor_dimensions,
    validate_tensors,
)


class LISTFoundationModelIT(LISTFoundationModelBackbone):
    listfmconfig: LISTFMConfig  # predefined
    vision_encoder: VisionEncoder  # predefined
    text_encoder: TextEncoder  # predefined
    tokenizer: SimpleTokenizer  # predefined
    bottleneck: Bottleneck  # predefined
    vision_decoder: VisionTextDecoder
    instruction_encoder: QwenInstructionEncoder

    def __init__(
        self,
        listfmconfig: LISTFMConfig,
        use_vision_decoder: bool,
    ) -> None:
        super().__init__(
            listfmconfig=listfmconfig,
            use_vision_decoder=use_vision_decoder,
        )
        if use_vision_decoder:
            self.vision_decoder = VisionTextDecoder(
                out_chans=listfmconfig.img_in_chan,
                feature_chans=listfmconfig.vision_enc_feat,
                decoder_feature_chans=listfmconfig.vision_dec_feat,
                num_pool_layers=listfmconfig.vision_enc_pool,
                image_width=listfmconfig.vision_img_w,
                instruction_dim=listfmconfig.text_enc_tf_w,
                input_chans=listfmconfig.img_in_chan,
            )
            self.instruction_encoder = QwenInstructionEncoder(
                model_name="Qwen/Qwen2.5-0.5B",
                proj_dim=listfmconfig.text_enc_tf_w,
                context_length=64,
            )

    def forward(
        self,
        img: Tensor,
        text: Tensor,
        grad_encoder: bool = True,
        use_bottleneck: bool = True,
        instruction: Tensor = None,
        flow_xt: Tensor | None = None,
        flow_t: Tensor | None = None,
    ) -> Tensor:
        validate_tensors([img, text])
        validate_tensor_dimensions([img], 4)
        validate_tensor_dimensions([text], 2)
        validate_tensor_channels(img, self.listfmconfig.img_in_chan)

        text_ctx = torch.no_grad() if not grad_encoder else nullcontext()
        text_full_feature = None
        text_context_length = self.text_encoder.context_length
        instruction_context_length = self.instruction_encoder.context_length

        def _pad_or_truncate(tokens: Tensor, context_length: int) -> Tensor:
            if tokens.shape[1] > context_length:
                return tokens[:, :context_length]
            if tokens.shape[1] < context_length:
                return F.pad(tokens, (0, context_length - tokens.shape[1]))
            return tokens

        text = _pad_or_truncate(text, text_context_length)
        if instruction is None:
            instruction_tokens = _pad_or_truncate(text, instruction_context_length)
            with text_ctx:
                (
                    _text_features,
                    text_full_feature,
                ) = self.instruction_encoder(
                    input_ids=instruction_tokens,
                )
            instruction = text_full_feature
        elif instruction.dtype in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            instruction = _pad_or_truncate(instruction, instruction_context_length)
            with text_ctx:
                (
                    _instruction_features,
                    instruction_full_feature,
                ) = self.instruction_encoder(
                    input_ids=instruction,
                )
            instruction = instruction_full_feature

        img = self._preprocess_image(img=img)
        if not grad_encoder:
            with torch.no_grad():
                img_full_feature = self.vision_encoder(x=img)
        else:
            img_full_feature = self.vision_encoder(x=img)

        img_decode = self.vision_decoder(
            x=img_full_feature,
            stack_feat=[],
            instruction=instruction,
            flow_xt=flow_xt,
            flow_t=flow_t,
        )

        return img_decode


def load_from_ckpt(
    ckpt_path: Path,
    from_scratch: bool = False,
    use_vision_decoder: bool = True,
    use_vision_decoder_weights: bool = True,
) -> LISTFoundationModelIT:
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
    model = LISTFoundationModelIT(
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
