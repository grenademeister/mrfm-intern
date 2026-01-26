import glob
import random
from dataclasses import dataclass
from enum import IntEnum

import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from datawrapper.simple_tokenizer import SimpleTokenizer
from datawrapper.undersampling import apply_fixed_mask
from datawrapper.warpper_utils import interpolate_to_target_width, resize_512

simple_tokenizer = SimpleTokenizer()

prob_half: float = 0.5


class DataKey(IntEnum):
    Input = 0
    Label = 1
    Text = 2
    Instruction = 3


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
    ):
        super().__init__()

        total_list: list[str] = []
        for _file_path in file_path:
            total_list += glob.glob(f"{_file_path}/{data_type}")

        self.file_list = total_list
        self.training_mode = training_mode

        if debug_mode:
            if training_mode:
                self.file_list = self.file_list[::1000]
            else:
                self.file_list = self.file_list[::5000]

        else:
            if training_mode:
                train_num = int(subject_num * slice_per_subject * train_percent)
                self.file_list = self.file_list[:train_num]
            else:
                # valid_num = int(subject_num * slice_per_subject * ((1 - train_percent) / 2))
                # self.file_list = self.file_list[:valid_num]
                self.file_list = self.file_list[:3000]

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

        # Augmentation
        if self.training_mode:
            if random.random() < prob_half:
                img = torch.flip(img, dims=[2])
            if random.random() < prob_half:
                img = torch.flip(img, dims=[1])

        img = interpolate_to_target_width(img, target_size=512)
        img = resize_512(img)

        input = img.clone()
        input, _, _ = apply_fixed_mask(input, acs_num=self.acs_num, parallel_factor=self.parallel_factor)
        input = input.abs().to(torch.float32)
        label = img.clone()

        text_token = simple_tokenizer.tokenize("", context_length=1536).squeeze()

        return (
            input,
            label,
            text_token,
        )

    def __len__(self) -> int:
        return len(self.file_list)


def get_data_wrapper_loader(
    file_path: list[str],
    training_mode: bool,
    loader_cfg: LoaderConfig,
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
    )

    _ = dataset[0]

    dataloader = DataLoader(
        dataset,
        batch_size=loader_cfg.batch,
        num_workers=loader_cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=loader_cfg.shuffle,
    )

    return (
        dataloader,
        dataset,
        len(dataset),
    )
