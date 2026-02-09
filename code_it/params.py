import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
from params_data import TEST_DATASET, TRAIN_DATASET, VALID_DATASET

default_run_dir: str = "/home/intern2/fm2026/fm_flow/logs"
RUN_DIR: str = os.environ.get("RUN_DIR", default_run_dir)


@dataclass
class GeneralConfig:
    # Dataset
    train_dataset: list[str] = field(default_factory=lambda: TRAIN_DATASET)
    valid_dataset: list[str] = field(default_factory=lambda: VALID_DATASET)
    test_dataset: list[str] = field(default_factory=lambda: TEST_DATASET)
    debugmode: bool = False
    data_type: str = "*.mat"

    # Logging
    log_lv: str = "INFO"
    run_dir: Path = Path(RUN_DIR)
    init_time: float = 0.0
    tb_enable: bool = True
    tb_log_dir: str = "tb"
    tb_log_batch: bool = True

    # Model experiment
    model_type: Literal["listfm_it", "unet"] = "listfm_it"
    text_encoding: Literal["llm", "clip"] = "llm"

    # Optimizer
    optimizer: Literal["adam", "adamw"] = "adamw"
    loss_model: Literal["l1", "l2"] = "l2"
    lr: float = 5e-4
    lr_decay: float = 0.95
    lr_tol: int = 1
    lr_schedule: Literal["exp", "cosine_warmup"] = "exp"
    lr_min: float = 1e-5
    lr_max_final: float = 1e-4
    lr_warmup_epochs: int = 2

    # Train params
    gpu: str = "0,1,2,3"
    train_batch: int = 16
    valid_batch: int = 128
    train_epoch: int = 80
    logging_density: int = 4
    valid_interval: int = 4
    valid_tol: int = 0
    num_workers: int = 32
    save_val: bool = True
    parallel: bool = True
    device: torch.device | None = None
    save_max_idx: int = 100

    # Pretrained
    pretrained: str = "/fast_storage/intern/code/share/checkpoint_3m.ckpt"
    use_bottleneck: bool = True
    grad_encoder: bool = True
    from_scratch: bool = False

    # Qwen instruction encoder
    # qwen_model_path: str = "/home/intern4/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
    qwen_model_path: str = "/home/intern4/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
    qwen_lora_path: str = "/home/intern4/fm2026/mrfm-intern/code_it_llm_ca/dapt/outputs/lora-0.5B/checkpoint-606"

    qwen_max_length: int = 128
    qwen_use_fast: bool = True
    qwen_trainable: bool = False

    # Rectified flow
    flow_t_min: float = 0.0
    flow_t_max: float = 1.0
    flow_eval_t: float = 0.9
    flow_noise_std: float = 1.0
    flow_eval_steps: int = 20
    flow_eval_eps: float = 1e-4

    # Data params
    acs_num: int = 24
    parallel_factor: int = 4
    subject_num: int = 3
    train_percent: float = 1.0
    slice_per_subject: int = 100

    tag: str = ""

    # Resume
    resume: bool = False
    resume_path: str = ""


@dataclass
class TestConfig:
    # Dataset
    trained_checkpoints: str = ""


# Argparser
parser = argparse.ArgumentParser(description="Training Configuration")
general_config_dict = asdict(GeneralConfig())
test_config_dict = asdict(TestConfig())

for key, default_value in general_config_dict.items():
    if isinstance(default_value, bool):
        parser.add_argument(
            f"--{key}",
            type=lambda x: x.lower() in ("true", "t", "yes", "y", "1"),
            default=None,
            help=f"Set {key} (true/false, default: {default_value})",
        )
    else:
        parser.add_argument(
            f"--{key}",
            type=type(default_value),
            default=None,
            help=f"Override for {key}",
        )


for key, default_value in test_config_dict.items():
    if isinstance(default_value, bool):
        parser.add_argument(
            f"--{key}",
            type=lambda x: x.lower() in ("true", "t", "yes", "y", "1"),
            default=None,
            help=f"Set {key} (true/false, default: {default_value})",
        )
    else:
        parser.add_argument(
            f"--{key}",
            type=type(default_value),
            default=None,
            help=f"Override for {key}",
        )

# Apply argparser
config = GeneralConfig()
args = parser.parse_args()

for key, value in vars(args).items():
    if value is not None and hasattr(config, key):
        if isinstance(getattr(config, key), bool):
            setattr(config, key, bool(value))
        else:
            setattr(config, key, value)

if config.text_encoding == "clip":
    config.qwen_model_path = ""
    config.qwen_lora_path = ""
    config.qwen_trainable = False
