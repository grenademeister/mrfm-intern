# code_it_new work log

## Summary
- Copied `code_it` to `code_it_new`.
- Replaced vision encoder with DINOv3 vits16plus + LoRA (all Linear layers), base weights frozen.
- Updated checkpoint loading to skip legacy vision encoder weights.
- Fixed bottleneck positional embedding to support dynamic token lengths.
- Replaced rectified-flow decoder with DiT-style decoder (same API).
- Ran sanity forward pass (offline HF cache).

## Sanity tests
- Forward pass with dummy tensors (CPU, offline HF cache): output shape `(1, 1, 512, 512)`.
- Notes: HF cache required; set `HF_HUB_OFFLINE=1` to avoid network.

## Model size and FLOPs (estimate)
**Config used for profiling** (representative, not from checkpoint):
- `img_in_chan=1`
- `vision_img_w=512`
- `vision_enc_feat=64`
- `vision_enc_pool=5`
- `text_enc_context=1536`
- `text_enc_tf_w=512`
- `bottleneck_width=1024`
- `clip_emb_dim=512`
- `vision_dec_feat=64`

**Parameter counts** (total, trainable):
- **TOTAL**: 271,667,716 (trainable 242,974,852)
- **VISION_ENCODER (DINOv3 + LoRA)**: 31,547,651 (trainable 2,854,787)
- **TEXT_ENCODER**: 64,478,209 (trainable 64,478,209)
- **BOTTLENECK**: 104,442,880 (trainable 104,442,880)
- **VISION_DECODER (DiT)**: 71,198,976 (trainable 71,198,976)

**FLOPs estimate (fvcore, B=1, 512x512)**:
- **~4.01e11 FLOPs**

**Important:** FLOPs are *under-counted* because fvcore does not support several ops used by transformers (e.g., `scaled_dot_product_attention`, `embedding`, etc.). The reported value is a lower bound.

## Notes
- DINOv3 weights downloaded to HF cache:
  `/home/grenade/.cache/huggingface/hub/models--facebook--dinov3-vits16plus-pretrain-lvd1689m/...`
- Use `DINOV3_MODEL_NAME` to override the backbone name.
- Use `HF_HUB_OFFLINE=1` to force offline loading.
