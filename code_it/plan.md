# Plan

## Goal
Create `code_it_new` with a DINOv3 backbone + LoRA adapter in the image path, keep text encoder, fuse via bottleneck, and decode with a larger DiT-based decoder. Validate tensor sizes and stability.

## Steps
1. Inventory current instruction-tuning model stack in `code_it` and how encoder/decoder are wired (model, config, training loop, loss).
2. Locate any pretraining or DiT decoder implementations in other directories to reuse.
3. Design the new image path:
   - DINOv3 backbone loading (weights, output tokens/feature map shape)
   - LoRA adapter placement (which layers, rank, scaling)
   - Projection to bottleneck width / expected encoder token dim
4. Design the new decoder path:
   - DiT-based decoder module
   - Conditioning inputs: fused image tokens + instruction features (and flow if required)
   - Output image channels and resolution
5. Create `code_it_new` by copying `code_it`, then implement new modules and wiring:
   - Config changes and model class updates
   - New encoder wrapper for DINOv3 + LoRA
   - New DiT decoder module
   - Update checkpoint loading logic and init
6. Update training/inference wiring to pass correct tensors (instruction, flow, etc.).
7. Add/extend shape checks to catch mismatches early.
8. Run unit-style shape tests (small batch, dummy inputs) to validate end-to-end forward.

## Tests (manual / script)
- Encoder shape test: input (B,C,H,W) → DINOv3 tokens/features, check expected token length and channel dim.
- Bottleneck fusion test: image tokens + text tokens produce fused tokens of same shape.
- Decoder shape test: fused tokens + instruction → output image (B,C,H,W).
- Full forward test: `LISTFoundationModelIT`-like forward path with dummy data.
- Optional: gradient flow test (ensure LoRA params receive gradients when encoder frozen).

## Open Questions / Assumptions
- DINOv3 model name, weights location, and API to load.
- Desired LoRA rank/alpha and which submodules to wrap.
- DiT decoder size (depth/width/patch size) and where to reuse from existing code.
- Whether flow-based inputs are still required for the new decoder.
