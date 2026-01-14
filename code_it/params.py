import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
from params_data import TEST_DATASET, TRAIN_DATASET, VALID_DATASET

default_run_dir: str = "/home/intern2/fm2026/code_intern/logs"
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

    # Model experiment
    model_type: Literal["listfm_it", "unet"] = "listfm_it"

    # Optimizer
    optimizer: Literal["adam", "adamw"] = "adam"
    loss_model: Literal["l1", "l2"] = "l2"
    lr: float = 1e-3
    lr_decay: float = 0.90
    lr_tol: int = 1

    # Train params
    gpu: str = "0,1"
    train_batch: int = 16
    valid_batch: int = 16
    train_epoch: int = 40
    logging_density: int = 8
    valid_interval: int = 4
    valid_tol: int = 2
    num_workers: int = 32
    save_val: bool = True
    parallel: bool = True
    device: torch.device | None = None
    save_max_idx: int = 100

    # Pretrained
    pretrained: str = "/home/juhyung/code/fm2026/code_downstream_recon/checkpoint_v2.1.ckpt"
    use_bottleneck: bool = True
    grad_encoder: bool = True
    from_scratch: bool = False

    # Data params
    acs_num: int = 24
    parallel_factor: int = 4
    subject_num: int = 3
    train_percent: float = 1.0
    slice_per_subject: int = 100

    tag: str = ""


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
