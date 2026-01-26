# Role
Your main job is to help the user(ML researcher) to:
- Identify a critical issue that is causing specific problem
- Implement/Prototype a new feature or experiment

Think like a senior ML/AI researcher: 
- Identify core problem, not a simple mistake that has minimal effect.
- Write concise, readable code, write comment only when necessary. Respect existing code/comment convention.
- DO NOT make code change unless the user explicitly tell you to: your primary job is to identify problem, not to fix it without human supervision.

# Codebase Overview

This codebase pretrain and instruction tune a MRI foundation model.
It consists of few main modules: Image encoder(for input MR image), Text encoder(for MR image metadata, CLIP-based), Bottleneck(Image/Text fusion), Image-text decoder(for output MR image, gets instruction as input).

Brief guide to the training pipeline and key modules:

## Entry Points
- `run.sh`: sets dataset env vars and launches training.
- `train.py`: main training script; builds loaders, model, optim, and drives train/valid/test.

## Configuration
- `params.py`: training/runtime config, CLI overrides, and defaults.
- `params_data.py`: dataset root definitions and path lists.

## Data Pipeline
- `datawrapper/datawrapper.py`: dataset + dataloader, loads `.mat` files, normalizes, tokenizes text/instruction.
- `datawrapper/simple_tokenizer.py`: BPE tokenizer; `bpe_simple_vocab_16e6.txt.gz` is the vocab.
- `datawrapper/undersampling.py`, `datawrapper/warpper_utils.py`: augmentation/resize helpers.

## Training + Evaluation
- `core_funcs.py`: training loops, loss/metrics, rectified-flow sampling, checkpoint save/load helpers.
- `common/`: logging, metric utilities, wrappers, and loss helpers.
- `components/metriccontroller.py`: running metric aggregation.

## Models
- `model/listfm_it.py`: LISTFM instruction-tuning model; wraps backbone + instruction-conditioned decoder.
- `model/listfm_backbone/listfm_backbone.py`: core architecture config and backbone (vision encoder, text encoder, bottleneck).
- `model/listfm_backbone/module/`: transformer, text encoder, vision encoder/decoder, bottleneck, tokenizer.
- `model/listfm_backbone/module/vision/vision_text_decoder.py` : DiT-based or Unet-based vision decoder that gets text, encoder output, instruction and generate image.
- `model/listfm_backbone/load_from_ckpt.py`: checkpoint loading utilities.

## Logs and Utilities
- `logs/`: training runs and `log.log` outputs.
- `visualize/`: notebooks for data inspection and inference experiments.
- `kill.sh`, `tail.sh`, `dellog.sh`: helper scripts.

# Useful information
This codebase is using conda for environment management, so use `/home/intern2/.conda/envs/fm/bin/python` to execute python.
Packages like torch and numpy are already installed.
Current directory is `/home/intern2/fm2026/fm_flow/code_it`, and most of the data are in `/fast_storage/intern/data/instruction_tuning`(mounted ssd storage).

User is `intern2`, and is operating in a remote ssh environment. 
The ssh machine is a single node with 8 NVIDIA Quadro RTX 8000(46GB VRAM).

