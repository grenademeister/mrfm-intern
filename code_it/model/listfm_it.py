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
    instruction_encoder: TextEncoder
    qwen_instruction_encoder: QwenInstructionEncoder | None

    def __init__(
        self,
        listfmconfig: LISTFMConfig,
        use_vision_decoder: bool,
        qwen_model_path: str | None = None,
        qwen_lora_path: str | None = None,
        qwen_trainable: bool = False,
        qwen_dtype: str = "bf16",
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
                block_type=listfmconfig.vision_block_type,
                instruction_dim=listfmconfig.clip_emb_dim,
                input_chans=listfmconfig.img_in_chan,
            )
            self.instruction_encoder = TextEncoder(
                embed_dim=listfmconfig.clip_emb_dim,
                context_length=64,
                vocab_size=listfmconfig.text_enc_vocab_size,
                transformer_width=listfmconfig.text_enc_tf_w,
                transformer_heads=listfmconfig.text_enc_tf_head,
                transformer_layers=listfmconfig.text_enc_tf_head,
                pretrained_model_weights=listfmconfig.text_enc_pretrained,
            )
        self.qwen_instruction_encoder = None
        if qwen_model_path:
            qwen_dtype_map = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }
            self.qwen_instruction_encoder = QwenInstructionEncoder(
                model_path=qwen_model_path,
                lora_path=qwen_lora_path,
                embed_dim=listfmconfig.clip_emb_dim,
                trainable=qwen_trainable,
                dtype=qwen_dtype_map.get(qwen_dtype, torch.bfloat16),
            )

    def forward(
        self,
        img: Tensor,
        text: Tensor,
        grad_encoder: bool = True,
        use_bottleneck: bool = True,
        instruction: Tensor = None,
        instruction_llm_ids: Tensor | None = None,
        instruction_llm_mask: Tensor | None = None,
        flow_xt: Tensor | None = None,
        flow_t: Tensor | None = None,
    ) -> Tensor:
        validate_tensors([img, text])
        validate_tensor_dimensions([img], 4)
        validate_tensor_dimensions([text], 2)
        validate_tensor_channels(img, self.listfmconfig.img_in_chan)

        instruction_copied = instruction
        instruction_copied = F.pad(
            instruction_copied,
            (0, self.listfmconfig.text_enc_context - instruction_copied.shape[1]),
        )

        text_ctx = torch.no_grad() if not grad_encoder else nullcontext()
        text_full_feature = None
        if self.qwen_instruction_encoder is None:
            if instruction is None:
                with text_ctx:
                    (
                        _text_features,
                        text_full_feature,
                    ) = self.instruction_encoder(
                        text=text,
                    )
                instruction = text_full_feature
            elif instruction.dtype in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                with text_ctx:
                    (
                        _instruction_features,
                        instruction_full_feature,
                    ) = self.instruction_encoder(
                        text=instruction,
                    )
                instruction = instruction_full_feature
        else:
            if instruction_llm_ids is None:
                raise ValueError("instruction_llm_ids must be provided when Qwen instruction encoder is enabled.")
            llm_ctx = nullcontext() if self.qwen_instruction_encoder.trainable else torch.no_grad()
            with llm_ctx:
                (
                    _llm_pooled,
                    instruction_full_feature,
                ) = self.qwen_instruction_encoder(
                    input_ids=instruction_llm_ids,
                    attention_mask=instruction_llm_mask,
                )
            instruction = instruction_full_feature

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
            if text_full_feature is None:
                with text_ctx:
                    (
                        _text_features,
                        text_full_feature,
                    ) = self.text_encoder(
                        text=instruction_copied,
                    )
            (
                img_full_feature,
                _text_full_feature,
            ) = self.bottleneck(
                vision_feature=img_full_feature,
                text_feature=text_full_feature,
            )

        img_decode = self.vision_decoder(
            x=img_full_feature,
            stack_feat=stack_feature,
            instruction=instruction,
            instruction_mask=instruction_llm_mask,
            flow_xt=flow_xt,
            flow_t=flow_t,
        )

        return img_decode


def load_from_ckpt(
    ckpt_path: Path,
    from_scratch: bool = False,
    use_vision_decoder: bool = True,
    use_vision_decoder_weights: bool = True,
    qwen_model_path: str | None = None,
    qwen_lora_path: str | None = None,
    qwen_trainable: bool = False,
    qwen_dtype: str = "bf16",
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
            "instruction_encoder",
            "qwen_instruction_encoder",
        ]:
            model_state_dict[new_key] = longitudinal_checkpoint_data["model_state_dict"][key]

        if use_vision_decoder_weights and new_key.split(".")[0] == "vision_decoder":
            model_state_dict[new_key] = longitudinal_checkpoint_data["model_state_dict"][key]

    # Initialize model
    model = LISTFoundationModelIT(
        listfmconfig,
        use_vision_decoder=use_vision_decoder,
        qwen_model_path=qwen_model_path,
        qwen_lora_path=qwen_lora_path,
        qwen_trainable=qwen_trainable,
        qwen_dtype=qwen_dtype,
    )

    # Load state dict
    if from_scratch:
        logger.warning("Loading model from scratch. All weights will be randomly initialized.")
    else:
        model.load_state_dict(model_state_dict, strict=False)
    model.qc()

    logger.success("Model state dict loaded successfully.")

    return model
