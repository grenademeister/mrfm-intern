import glob
import random
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from datawrapper.simple_tokenizer import SimpleTokenizer
from datawrapper.undersampling import apply_fixed_mask
from datawrapper.warpper_utils import interpolate_to_target_width, resize_512

simple_tokenizer = SimpleTokenizer()
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", use_fast=True)
if qwen_tokenizer.pad_token is None:
    qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

prob_half: float = 0.5
norm_eps: float = 1e-6


def _coerce_matlab_text(value: object) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.item()
        else:
            value = value.flatten()
            if value.dtype.kind in {"U", "S"}:
                value = "".join(str(v) for v in value)
            else:
                value = str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mean = tensor.mean()
    std = tensor.std(unbiased=False).clamp_min(norm_eps)
    return (tensor - mean) / std


class DataKey(IntEnum):
    Input = 0
    Label = 1
    Text = 2
    Instruction = 3
    TaskName = 4


@dataclass
class LoaderConfig:
    batch: int
    num_workers: int
    shuffle: bool
    debug_mode: bool
    acs_num: int
    parallel_factor: int
    data_type: str
    subject_num: int
    train_percent: float
    slice_per_subject: int


class DataWrapper(Dataset):
    num_timesteps: int
    file_list: list[str]
    training_mode: bool
    acs_num: int
    parallel_factor: int
    data_type: str
    subject_num: int
    train_percent: float
    slice_per_subject: int
    file_path_list: list[str]

    def __init__(
        self,
        file_path: list[str],
        training_mode: bool,
        debug_mode: bool,
        acs_num: int,
        parallel_factor: int,
        data_type: str,
        subject_num: int,
        train_percent: float,
        slice_per_subject: int,
        split: str,
    ):
        super().__init__()

        total_list: list[str] = []
        self.file_path_list = file_path
        
        for _file_path in file_path:
            files = glob.glob(f"{_file_path}/{data_type}")
            if debug_mode:
                if split == "train":
                    files = files[:1280]
                else:
                    files = files[:64]
            else:
                if split == "train":
                    files = files[:10000]
                else:
                    files = files[:200]
            
            total_list += files

        self.file_list = total_list
        self.training_mode = training_mode

        self.acs_num = acs_num
        self.parallel_factor = parallel_factor

        print(f"DataWrapper initialized with {len(self.file_list)} samples.")
        print(f"Working directory: {file_path}")

    def __getitem__(
        self,
        idx: int,
    ):
        np_data = loadmat(self.file_list[idx])["image"]
        img = torch.from_numpy(np_data).unsqueeze(0).to(torch.float32)  # (1, H, W)

        tgt = loadmat(self.file_list[idx])["label"]
        tgt = torch.from_numpy(tgt).unsqueeze(0).to(torch.float32)

        # Augmentation
        if self.training_mode:
            if random.random() < prob_half:
                img = torch.flip(img, dims=[2])
                tgt = torch.flip(tgt, dims=[2])
            if random.random() < prob_half:
                img = torch.flip(img, dims=[1])
                tgt = torch.flip(tgt, dims=[1])

        img = interpolate_to_target_width(img, target_size=512)
        img = resize_512(img)
        tgt = resize_512(interpolate_to_target_width(tgt, target_size=512))

        img = _normalize_tensor(img)
        tgt = _normalize_tensor(tgt)

        input = img.clone()
        # input, _, _ = apply_fixed_mask(input, acs_num=self.acs_num, parallel_factor=self.parallel_factor)

        text = loadmat(self.file_list[idx])["text"][0][0]
        text = _coerce_matlab_text(text)
        text_token = simple_tokenizer.tokenize(text, context_length=1536).squeeze()

        instruction = loadmat(self.file_list[idx])["instruction"][0][0]
        instruction = _coerce_matlab_text(instruction)
        instruction_enc = qwen_tokenizer(
            instruction,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        instruction_token = instruction_enc["input_ids"].squeeze(0)

        # Extract task_name from file path
        current_file = self.file_list[idx]
        task_name = "unknown"
        for base_path in self.file_path_list:
            if current_file.startswith(base_path):
                task_name = base_path.rstrip('/').split('/')[-2]
                break

        return (
            input,
            tgt,
            text_token,
            instruction_token,
            task_name,
        )

    def __len__(self) -> int:
        return len(self.file_list)


def get_data_wrapper_loader(
    file_path: list[str],
    training_mode: bool,
    loader_cfg: LoaderConfig,
    split: str,
) -> tuple[
    DataLoader,
    DataWrapper,
    int,
]:
    dataset = DataWrapper(
        file_path=file_path,
        training_mode=training_mode,
        debug_mode=loader_cfg.debug_mode,
        acs_num=loader_cfg.acs_num,
        parallel_factor=loader_cfg.parallel_factor,
        data_type=loader_cfg.data_type,
        subject_num=loader_cfg.subject_num,
        train_percent=loader_cfg.train_percent,
        slice_per_subject=loader_cfg.slice_per_subject,
        split=split,
    )

    _ = dataset[0]

    dataloader = DataLoader(
        dataset,
        batch_size=loader_cfg.batch,
        num_workers=loader_cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=loader_cfg.shuffle,
        drop_last=True,
    )

    return (
        dataloader,
        dataset,
        len(dataset),
    )
