"""
#  Copyright
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email :
#
"""

from dataclasses import asdict
from pathlib import Path

import torch

from .listfm_backbone import LISTFMConfig, LISTFoundationModelBackbone
from .utils import logger


def load_from_ckpt(  # noqa: C901
    ckpt_path: Path,
    use_vision_decoder: bool = False,
    use_vision_decoder_weights: bool = False,
) -> LISTFoundationModelBackbone:
    """
    Load a LISTFoundationModelBackbone from a checkpoint file.

    Args:
        ckpt_path (Path): Path to the checkpoint file.
        use_vision_decoder (bool): Whether to use the vision decoder module.
        use_vision_decoder_weights (bool): Whether to load weights for the vision decoder module.

    Returns:
        LISTFoundationModelBackbone: The loaded model.
    """

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

    logger.debug("Checkpoint loaded successfully.")

    # Load model config
    listfmconfig = LISTFMConfig.from_dict(longitudinal_checkpoint_data["model_config"])
    logger.debug("Model Config")
    config_dict = asdict(listfmconfig)
    for k in config_dict:
        logger.debug(f"{k:<30}:{config_dict[k]}")

    # Load model state dict
    model_state_dict: dict[str, torch.Tensor] = {}
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

    for k in model_state_dict:
        logger.trace(f"Loaded weight: {k}")

    # Initialize model
    model = LISTFoundationModelBackbone(
        listfmconfig=listfmconfig,
        use_vision_decoder=use_vision_decoder,
    )

    # Load state dict
    if use_vision_decoder_weights:
        model.load_state_dict(model_state_dict, strict=True)
    else:
        model.load_state_dict(model_state_dict, strict=False)

    logger.debug("Model state dict loaded successfully.")

    return model
