"""
#  Copyright
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email :
#
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from .module import (
    BlockType,
    Bottleneck,
    SimpleTokenizer,
    TextEncoder,
    VisionDecoder,
    VisionEncoder,
)
from .utils import (
    logger,
    validate_tensor_channels,
    validate_tensor_dimensions,
    validate_tensors,
)


@dataclass
class LISTFMConfig:
    """Configuration container for LIST foundation model.

    Fields document expected types and constraints.

    Attributes
    ----------
    img_in_chan : int
        Number of input image channels (e.g., 3 for RGB)
    vision_enc_feat : int
        Base number of feature channels in vision encoder
    vision_enc_pool : int
        Number of pooling/scale reductions in vision encoder
    vision_enc_tf_layer : int
        Number of transformer layers in vision encoder
    vision_enc_tf_head : int
        Number of attention heads in vision encoder transformer
    vision_img_w : int
        Target square image width for the model
    vision_block_type : BlockType
        Block type enumerator for vision blocks
    text_enc_context : int
        Max text context length (number of tokens)
    text_enc_vocab_size : int
        Vocabulary size for tokenizer/text encoder
    text_enc_tf_w : int
        Transformer width for text encoder
    text_enc_tf_layer : int
        Number of transformer layers for text encoder
    text_enc_tf_head : int
        Transformer heads for text encoder
    text_enc_pretrained : Optional[Path]
        Optional path to pretrained text encoder weights
    bottleneck_width : int
        Bottleneck projection width
    bottleneck_layer : int
        Number of layers in bottleneck
    bottleneck_head : int
        Number of heads in bottleneck attention
    tokenizer_bpe : Path
        Path to BPE tokenizer file
    clip_emb_dim : int
        Final embedding dimension for CLIP-like output
    """

    img_in_chan: int  # 1
    vision_enc_feat: int  # 64
    vision_enc_pool: int  # 4
    vision_enc_tf_layer: int  # 12
    vision_enc_tf_head: int  # 8
    vision_img_w: int  # 512
    vision_block_type: BlockType  # 'block3'
    text_enc_context: int  # 1536
    text_enc_vocab_size: int  # 49408
    text_enc_tf_w: int  # 512
    text_enc_tf_layer: int  # 12
    text_enc_tf_head: int  # 8
    text_enc_pretrained: Path | None  # None
    bottleneck_width: int  # 512
    bottleneck_layer: int  # 12
    bottleneck_head: int  # 8
    tokenizer_bpe: Path  # PosixPath('bpe_simple_vocab_16e6.txt.gz')
    clip_emb_dim: int  # 512
    vision_dec_feat: int = 16

    def to_dict(self) -> dict:
        d = asdict(self)
        # convert Paths to str
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        # remove internal attributes
        d.pop("_base_path", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LISTFMConfig":
        d = dict(d)
        if "_base_path" in d:
            d.pop("_base_path")
        # convert some paths
        if "tokenizer_bpe" in d:
            d["tokenizer_bpe"] = Path(d["tokenizer_bpe"])
        if "text_enc_pretrained" in d and d["text_enc_pretrained"] is not None:
            d["text_enc_pretrained"] = Path(d["text_enc_pretrained"])
        return cls(**d)


class LISTFoundationModelBackbone(torch.nn.Module):
    """
    LIST Foundation Model Backbone
    Attributes:
    linstfmconfig: LISTFMConfig
    vision_encoder: VisionEncoder
    text_encoder: TextEncoder
    tokenizer: SimpleTokenizer
    bottleneck: Bottleneck
    vision_decoder: VisionDecoder | None

    Methods:
    _parameter_check: callable
    _pad_square: callable
    _resize_target_width: callable
    _preprocess_image: callable
    encode_image: callable
    encode_text: callable
    fuse_modalities: callable
    forward: callable
    qc: callable
    """

    listfmconfig: LISTFMConfig
    vision_encoder: VisionEncoder
    text_encoder: TextEncoder
    tokenizer: SimpleTokenizer
    bottleneck: Bottleneck
    vision_decoder: VisionDecoder | None

    def __init__(
        self,
        listfmconfig: LISTFMConfig,
        use_vision_decoder: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(listfmconfig, LISTFMConfig):
            raise TypeError("listfmconfig must be an instance of LISTFMConfig")

        self._parameter_check(listfmconfig=listfmconfig)

        self.listfmconfig = listfmconfig

        self.vision_encoder = VisionEncoder(
            model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
            finetune_encoder=True,
            input_size=None,
            input_is_normalized=True,
        )

        self.text_encoder = TextEncoder(
            embed_dim=listfmconfig.clip_emb_dim,
            context_length=listfmconfig.text_enc_context,
            vocab_size=listfmconfig.text_enc_vocab_size,
            transformer_width=listfmconfig.text_enc_tf_w,
            transformer_layers=listfmconfig.text_enc_tf_layer,
            transformer_heads=listfmconfig.text_enc_tf_head,
            pretrained_model_weights=listfmconfig.text_enc_pretrained,
        )

        self.tokenizer = SimpleTokenizer(
            bpe_path=listfmconfig.tokenizer_bpe,
        )

        context_length = listfmconfig.vision_enc_feat * (2 ** (listfmconfig.vision_enc_pool)) + listfmconfig.text_enc_context + 1

        self.bottleneck = Bottleneck(
            width=listfmconfig.bottleneck_width,
            layers=listfmconfig.bottleneck_layer,
            heads=listfmconfig.bottleneck_head,
            context_length=context_length,
        )

        if use_vision_decoder:
            self.vision_decoder = VisionDecoder(
                out_chans=listfmconfig.img_in_chan,
                feature_chans=listfmconfig.vision_enc_feat,
                num_pool_layers=listfmconfig.vision_enc_pool,
                block_type=listfmconfig.vision_block_type,
            )
        else:
            self.vision_decoder = None

    def _parameter_check(
        self,
        listfmconfig: LISTFMConfig,
    ) -> None:
        vision_w = listfmconfig.vision_enc_feat * (2 ** (listfmconfig.vision_enc_pool - 1))
        if vision_w == listfmconfig.text_enc_tf_w == listfmconfig.bottleneck_width:
            logger.debug("Width check success")
        else:
            if vision_w <= listfmconfig.bottleneck_width and listfmconfig.text_enc_tf_w <= listfmconfig.bottleneck_width:
                logger.debug("Width check success")
            else:
                raise ValueError(f"Width check failed :{vision_w}, {listfmconfig.text_enc_tf_w}, {listfmconfig.bottleneck_width}")
        if (
            vision_w % listfmconfig.vision_enc_tf_head
            == listfmconfig.text_enc_tf_w % listfmconfig.text_enc_tf_head
            == listfmconfig.bottleneck_width % listfmconfig.bottleneck_head
        ):
            logger.debug("Head check success")
        else:
            raise ValueError("Witch % Head had to be 0")
        if listfmconfig.tokenizer_bpe:
            bpe_path = listfmconfig.tokenizer_bpe
            if not os.path.isabs(bpe_path):
                module_bpe = os.path.join(
                    os.path.dirname(__file__),
                    "module",
                    "tokenizer",
                    bpe_path,
                )
                repo_bpe = os.path.join(os.getcwd(), bpe_path)
                if os.path.exists(module_bpe):
                    bpe_path = module_bpe
                elif os.path.exists(repo_bpe):
                    bpe_path = repo_bpe
            if os.path.exists(bpe_path):
                logger.debug("BPE file exists.")
                listfmconfig.tokenizer_bpe = bpe_path
            else:
                raise FileNotFoundError(f"BPE file not found: {listfmconfig.tokenizer_bpe}")

        if listfmconfig.text_enc_pretrained:
            pretrained_path = listfmconfig.text_enc_pretrained
            if not os.path.isabs(pretrained_path):
                module_pretrained = os.path.join(os.path.dirname(__file__), "module", pretrained_path)
                repo_pretrained = os.path.join(os.getcwd(), pretrained_path)
                if os.path.exists(module_pretrained):
                    pretrained_path = module_pretrained
                elif os.path.exists(repo_pretrained):
                    pretrained_path = repo_pretrained
            if os.path.exists(pretrained_path):
                logger.debug("Text encoder pretrained model file exists.")
                listfmconfig.text_enc_pretrained = pretrained_path
            else:
                raise FileNotFoundError(
                    f"Text encoder pretrained model file not found: {listfmconfig.text_enc_pretrained}"
                )

    @staticmethod
    def _pad_square(
        img: Tensor,
        target_size: int | None = None,
    ) -> Tensor:
        img_dims = img.dim()
        if img_dims not in {4, 3}:
            raise ValueError("Input image must be 3D or 4D tensor.")

        if img_dims == 3:
            img = img.unsqueeze(0)

        _, _, h, w = img.shape
        size = max(h, w)
        if target_size is not None:
            size = max(size, target_size)

        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        padding = (pad_w, size - w - pad_w, pad_h, size - h - pad_h)
        img = torch.nn.functional.pad(img, padding, mode="constant", value=0)

        if img_dims == 3:
            img = img.squeeze(0)

        return img

    def _resize_target_width(
        self,
        img: Tensor,
    ) -> Tensor:
        img_dims = img.dim()
        if img_dims not in {4, 3}:
            raise ValueError("Input image must be 3D or 4D tensor.")

        if img_dims == 3:
            img = img.unsqueeze(0)

        target_size = self.listfmconfig.vision_img_w

        _, _, h, w = img.shape
        if h == target_size and w == target_size:
            if img_dims == 3:
                img = img.squeeze(0)
            return img

        if h > target_size or w > target_size:
            img = torch.nn.functional.interpolate(
                img,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )

        else:
            img = self._pad_square(img, target_size=target_size)

        if img_dims == 3:
            img = img.squeeze(0)

        return img

    def _preprocess_image(
        self,
        img: Tensor,
    ) -> Tensor:
        validate_tensors([img])
        validate_tensor_dimensions([img], 4)
        validate_tensor_channels(img, self.listfmconfig.img_in_chan)

        img = self._resize_target_width(img=img)
        return img

    def encode_image(
        self,
        img: Tensor,
    ) -> tuple[Tensor, list[Tensor]]:
        """Encode image alone and return (img_full_feature, stack_feature)

        img: (B, C, H, W)
        Returns img_full_feature: Tensor (B, N, D), stack_feature: list[Tensor]
        """
        img = self._preprocess_image(img)
        img_full_feature = self.vision_encoder(x=img)

        validate_tensor_dimensions([img_full_feature], 4)
        return img_full_feature, []

    def encode_text(
        self,
        text: Tensor,
    ) -> Tensor:
        """Encode text alone and return text_full_feature.

        text: (B, L) int tokens
        Returns text_full_feature: Tensor (B, L, D)
        """
        validate_tensors([text])
        validate_tensor_dimensions([text], 2)
        _, text_full_feature = self.text_encoder(text=text)
        validate_tensor_dimensions([text_full_feature], 3)
        return text_full_feature

    def fuse_modalities(
        self,
        img_feat: Tensor,
        text_feat: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the bottleneck fusion stage and return fused features.
        Both inputs expected to be 3D tensors.
        """
        validate_tensors([img_feat, text_feat])
        validate_tensor_dimensions([img_feat, text_feat], 3)
        img_fused, text_fused = self.bottleneck(vision_feature=img_feat, text_feature=text_feat)
        validate_tensor_dimensions([img_fused, text_fused], 3)
        return img_fused, text_fused

    def decode_image(
        self,
        img_feat: torch.Tensor,
        stack: list[torch.Tensor],
        flow_xt: torch.Tensor | None = None,
        flow_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.vision_decoder is None:
            raise RuntimeError("Vision decoder not enabled.")
        if img_feat.dim() != 3:
            raise ValueError("VisionDecoder expects 3D (B, N, D) features.")
        if flow_xt is None:
            flow_xt = torch.randn(
                img_feat.shape[0],
                self.listfmconfig.img_in_chan,
                self.listfmconfig.vision_img_w,
                self.listfmconfig.vision_img_w,
                device=img_feat.device,
            )
        if flow_t is None:
            flow_t = torch.full((img_feat.shape[0], 1), 0.5, device=img_feat.device)
        return self.vision_decoder.forward(
            x=img_feat,
            stack_feat=stack,
            flow_xt=flow_xt,
            flow_t=flow_t,
        )

    def inference(
        self,
        img: Tensor,
        text: Tensor,
        use_backbone: bool = True,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Main forward.

        Args:
            img: (B, C, H, W)
            text: (B, L)
            use_amp: if True, run encoder attention under autocast for fp16 speedups

        Returns:
            img_full_feature, text_full_feature, stack_feature
        """

        validate_tensors([img, text])
        validate_tensor_dimensions([img], 4)
        validate_tensor_dimensions([text], 2)
        validate_tensor_channels(img, self.listfmconfig.img_in_chan)

        img = self._preprocess_image(img=img)

        img_full_feature, stack_feature = self.encode_image(
            img=img,
        )
        text_full_feature = self.encode_text(
            text=text,
        )

        if use_backbone and img_full_feature.dim() == 3:
            img_full_feature, text_full_feature = self.fuse_modalities(
                img_feat=img_full_feature,
                text_feat=text_full_feature,
            )

        return (
            img_full_feature,
            text_full_feature,
            stack_feature,
        )

    def qc(self) -> None:
        logger.debug("QC start.")
        img = torch.ones(
            2,
            self.listfmconfig.img_in_chan,
            self.listfmconfig.vision_img_w,
            self.listfmconfig.vision_img_w,
        )  # (B, 1, 512, 512)
        text = self.tokenizer.tokenize(
            texts=["a dog", "a cat"],
            context_length=self.listfmconfig.text_enc_context,
        )
        logger.debug(f"Image size: {img.shape}")  # (B, 1, 512, 512)
        logger.debug(f"Text size: {text.shape}")  # (B, 1536)
        (
            img_full_feature,
            text_full_feature,
            _stack_feature,
        ) = self.inference(img=img, text=text)
        logger.debug(f"img_full_feature size: {img_full_feature.shape}")
        logger.debug(f"text_full_feature size: {text_full_feature.shape}")
        logger.debug(
            f"img mean std max min: {img_full_feature.mean().item():.4f} {img_full_feature.std().item():.4f} {img_full_feature.max().item():.4f} {img_full_feature.min().item():.4f}"  # noqa: E501
        )
        logger.debug(
            f"text mean std max min: {text_full_feature.mean().item():.4f} {text_full_feature.std().item():.4f} {text_full_feature.max().item():.4f} {text_full_feature.min().item():.4f}"  # noqa: E501
        )
        for i, stack_f in enumerate(_stack_feature):
            logger.debug(f"stack_feature[{i}] size: {stack_f.shape}")

        if self.vision_decoder is not None and img_full_feature.dim() == 3:
            recon_img = self.decode_image(
                img_feat=img_full_feature,
                stack=_stack_feature.copy(),
            )
            logger.debug(f"recon_img size: {recon_img.shape}")
            if recon_img.shape == img.shape:
                logger.debug("Reconstructed image size matches input image size.")
            else:
                raise ValueError("Reconstructed image size does not match input image size.")

        logger.debug("QC success.")
