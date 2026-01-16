import math
import os
import time
from collections.abc import Callable
from dataclasses import asdict
from enum import Enum
from pathlib import Path

import torch
from scipy.io import savemat
from torch import Tensor
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from common.logger import logger
from common.metric import calculate_psnr, calculate_ssim
from common.utils import seconds_to_dhms
from components.metriccontroller import MetricController
from datawrapper.datawrapper import DataKey
from model.listfm_it import LISTFoundationModelIT
from params import config

NETWORK = LISTFoundationModelIT | torch.nn.DataParallel[LISTFoundationModelIT]
OPTIM = Adam | AdamW


class ModelType(str, Enum):
    LISTFM_IT = "listfm_it"
    UNET = "unet"

    @classmethod
    def from_string(cls, value: str) -> "ModelType":
        try:
            return cls(value)
        except ValueError as err:
            raise ValueError(f"Invalid ModelType value: {value}. Must be one of {list(cls)} : {err}") from err


def get_optim(
    network: NETWORK | None,
    optimizer: str,
) -> OPTIM | None:
    if network is None:
        return None
    if optimizer == "adam":
        return Adam(network.parameters(), betas=(0.9, 0.99))
    elif optimizer == "adamw":
        return AdamW(network.parameters(), betas=(0.9, 0.99), weight_decay=0.0)
    else:
        raise KeyError("optimizer not matched")


def get_loss_func(
    loss_model: str,
) -> Callable:
    if loss_model == "l1":
        return torch.nn.L1Loss(reduction="none")
    elif loss_model == "l2":
        return torch.nn.MSELoss(reduction="none")
    else:
        raise KeyError("loss func not matched")


def sample_flow_t(
    batch: int,
    device: torch.device,
) -> Tensor:
    t = torch.rand(batch, 1, 1, 1, device=device)
    if config.flow_t_min != 0.0 or config.flow_t_max != 1.0:
        t = t * (config.flow_t_max - config.flow_t_min) + config.flow_t_min
    return t


def make_flow_pair(
    label: Tensor,
    t: Tensor,
) -> tuple[Tensor, Tensor]:
    x0 = torch.randn_like(label) * config.flow_noise_std
    xt = (1.0 - t) * x0 + t * label
    return x0, xt


def _rectified_flow_sample_dep(
    model: NETWORK,
    img: Tensor,
    text: Tensor,
    instruction: Tensor,
) -> Tensor:
    """Deprecated rectified flow sampling function."""
    steps = max(1, int(config.flow_eval_steps))
    t_vals = torch.linspace(1.0, config.flow_eval_eps, steps, device=img.device)
    x = torch.randn_like(img) * config.flow_noise_std

    for i in range(steps - 1):
        t = t_vals[i].view(1, 1, 1, 1).expand(img.shape[0], 1, 1, 1)
        pred = model.forward(
            img=img,
            text=text,
            use_bottleneck=config.use_bottleneck,
            grad_encoder=config.grad_encoder,
            instruction=instruction,
            flow_xt=x,
            flow_t=t.view(img.shape[0], 1),
        )
        t_safe = torch.clamp(t, min=config.flow_eval_eps)
        v = (x - pred) / t_safe
        dt = (t_vals[i + 1] - t_vals[i]).view(1, 1, 1, 1)
        x = x + dt * v

    t_last = t_vals[-1].view(1, 1, 1, 1).expand(img.shape[0], 1, 1, 1)
    pred = model.forward(
        img=img,
        text=text,
        use_bottleneck=config.use_bottleneck,
        grad_encoder=config.grad_encoder,
        instruction=instruction,
        flow_xt=x,
        flow_t=t_last.view(img.shape[0], 1),
    )
    t_safe = torch.clamp(t_last, min=config.flow_eval_eps)
    v = (x - pred) / t_safe
    x = x + (0.0 - t_last) * v
    return x


def rectified_flow_sample(
    model: NETWORK,
    img: Tensor,
    text: Tensor,
    instruction: Tensor,
    steps: int | None = None,
    t_eps: float | None = None,
) -> Tensor:
    steps = max(1, int(40 if steps is None else steps))
    t_eps = config.flow_eval_eps if t_eps is None else t_eps
    s = torch.linspace(0.0, 1.0, steps + 1, device=img.device)
    t_vals = 0.5 - 0.5 * torch.cos(math.pi * s)
    z = torch.randn_like(img) * config.flow_noise_std

    for i in range(steps):
        t = t_vals[i].view(1, 1, 1, 1).expand(img.shape[0], 1, 1, 1)
        t_next = t_vals[i + 1].view(1, 1, 1, 1).expand(img.shape[0], 1, 1, 1)
        x_pred = model.forward(
            img=img,
            text=text,
            use_bottleneck=config.use_bottleneck,
            grad_encoder=config.grad_encoder,
            instruction=instruction,
            flow_xt=z,
            flow_t=t.view(img.shape[0], 1),
        )
        denom = (1.0 - t).clamp_min(t_eps)
        v_pred = (x_pred - z) / denom
        z = z + (t_next - t) * v_pred

    return z


def get_learning_rate(
    epoch: int,
    lr: float,
    lr_decay: float,
    lr_tol: int,
) -> float:
    if config.lr_schedule == "exp":
        factor = epoch - lr_tol if lr_tol < epoch else 0
        return lr * (lr_decay**factor)
    if config.lr_schedule != "cosine_warmup":
        raise KeyError(f"lr_schedule not matched: {config.lr_schedule}")

    min_lr = config.lr_min
    max_lr = lr
    max_lr_final = config.lr_max_final
    warmup_epochs = max(0, int(config.lr_warmup_epochs))
    total_epochs = max(1, int(config.train_epoch))

    if warmup_epochs > 0 and epoch < warmup_epochs:
        warmup_progress = (epoch + 1) / warmup_epochs
        return min_lr + (max_lr - min_lr) * warmup_progress

    steps_after_warmup = max(1, total_epochs - warmup_epochs - 1)
    progress = (epoch - warmup_epochs) / steps_after_warmup
    progress = min(1.0, max(0.0, progress))

    current_max = max_lr + (max_lr_final - max_lr) * progress
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (current_max - min_lr) * cosine


def set_optimizer_lr(
    optimizer: OPTIM | None,
    learning_rate: float,
) -> OPTIM | None:
    if optimizer is None:
        return None
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
    return optimizer


def log_summary(
    init_time: float,
    state: MetricController,
    log_std: bool = False,
) -> None:
    spend_time = seconds_to_dhms(time.time() - init_time)
    for key in state.state_dict:
        if log_std:
            summary = f"{spend_time} | {key}: {state.mean(key):0.3e} + {state.std(key):0.3e} "
            logger.info(summary)
        else:
            summary = f"{spend_time} | {key}: {state.mean(key):0.3e}"
            logger.info(summary)


def save_checkpoint(
    network: NETWORK,
    run_dir: Path,
    epoch: str | int | None = None,
) -> None:
    if epoch is None:
        epoch = "best"
    os.makedirs(run_dir / "checkpoints", exist_ok=True)
    if ModelType.from_string(config.model_type) == ModelType.LISTFM_IT:
        torch.save(
            {
                "model_state_dict": network.state_dict(),
                "model_config": asdict(network.module.listfmconfig if isinstance(network, torch.nn.DataParallel) else network.listfmconfig),
            },
            run_dir / f"checkpoints/checkpoint_{epoch}.ckpt",
        )
    elif ModelType.from_string(config.model_type) == ModelType.UNET:
        torch.save(
            {
                "model_state_dict": network.state_dict(),
            },
            run_dir / f"checkpoints/checkpoint_{epoch}.ckpt",
        )
    else:
        raise TypeError("network type not supported")


def zero_optimizers(
    optim_list: list[OPTIM | None],
) -> None:
    for opt in optim_list:
        if opt is not None:
            opt.zero_grad()


def step_optimizers(
    optim_list: list[OPTIM | None],
) -> None:
    for opt in optim_list:
        if opt is not None:
            opt.step()


def save_result_to_mat(
    test_dir: Path,
    batch_cnt: int,
    tesner_dict: dict[str, Tensor | None],
    img_cnt: int,
) -> None:
    os.makedirs(test_dir, exist_ok=True)
    save_dict = {}

    if batch_cnt == 0:
        logger.warning("batch_cnt is 0, no data to save")
        return

    for i in range(batch_cnt):
        for key, value in tesner_dict.items():
            if value is not None:
                save_dict[key] = value.cpu().detach().numpy()[i, ...]

        idx = img_cnt + i + 1
        savemat(f"{test_dir}/{idx}_res.mat", save_dict)


def train_epoch_listfm_vision_pretraining(
    _data: dict[DataKey, Tensor | str],
    network: NETWORK,
    epoch: int,
    train_state: MetricController,
) -> int:
    loss_func = get_loss_func(config.loss_model)

    input: Tensor = _data[DataKey.Input].to(config.device)
    text: Tensor = _data[DataKey.Text].to(config.device)
    label: Tensor = _data[DataKey.Label].to(config.device)
    instruction: Tensor = _data[DataKey.Instruction].to(config.device)
    img_cnt_minibatch = input.shape[0]

    flow_t = sample_flow_t(batch=img_cnt_minibatch, device=config.device)
    flow_x0, flow_xt = make_flow_pair(label=label, t=flow_t)

    output = network.forward(
        img=input,
        text=text,
        instruction=instruction,
        use_bottleneck=config.use_bottleneck,
        grad_encoder=config.grad_encoder,
        flow_xt=flow_xt,
        flow_t=flow_t.view(img_cnt_minibatch, 1),
    )

    loss = torch.mean(loss_func(output, label), dim=(1, 2, 3), keepdim=True)

    torch.mean(loss).backward()
    train_state.add("loss", loss)

    return img_cnt_minibatch


def train_epoch(
    train_loader: DataLoader,
    train_len: int,
    network: NETWORK,
    optim_list: list[OPTIM | None],
    epoch: int,
) -> None:
    train_state = MetricController()
    train_state.reset()
    network.train()

    logging_cnt: int = 1
    img_cnt: int = 0
    for _data in train_loader:
        zero_optimizers(optim_list=optim_list)
        img_cnt_minibatch = train_epoch_listfm_vision_pretraining(
            _data=_data,
            network=network,
            epoch=epoch,
            train_state=train_state,
        )

        step_optimizers(optim_list=optim_list)
        img_cnt += img_cnt_minibatch
        if img_cnt > (train_len / config.logging_density * logging_cnt):
            log_summary(init_time=config.init_time, state=train_state)
            logging_cnt += 1

    log_summary(init_time=config.init_time, state=train_state)


def test_part_listfm_vision_pretraining(
    _data: dict[DataKey, Tensor | str],
    test_dir: Path,
    model: NETWORK,
    save_val: bool,
    test_state: MetricController,
    img_cnt: int,
) -> float:
    loss_func = get_loss_func(config.loss_model)

    input: Tensor = _data[DataKey.Input].to(config.device)
    text: Tensor = _data[DataKey.Text].to(config.device)
    label: Tensor = _data[DataKey.Label].to(config.device)
    instruction: Tensor = _data[DataKey.Instruction].to(config.device)

    batch_cnt = input.shape[0]
    with torch.no_grad():
        output = rectified_flow_sample(
            model=model,
            img=input,
            text=text,
            instruction=instruction,
        )

    loss = torch.mean(loss_func(output, label), dim=(1, 2, 3), keepdim=True)

    test_state.add("loss", loss)

    test_state.add("psnr", calculate_psnr(output, label))
    test_state.add("ssim", calculate_ssim(output, label))

    if save_val:
        save_result_to_mat(
            test_dir=test_dir,
            batch_cnt=batch_cnt,
            tesner_dict={
                "input": input,
                "out": output,
                "label": label,
            },
            img_cnt=img_cnt,
        )

    return batch_cnt


def test_part(
    epoch: int,
    data_loader: DataLoader,
    network: NETWORK,
    run_dir: Path,
    save_val: bool,
) -> float:
    test_state = MetricController()
    test_state.reset()
    network.eval()
    model = network.module if isinstance(network, torch.nn.DataParallel) else network

    img_cnt: int = 0
    for _data in data_loader:
        batch_cnt = test_part_listfm_vision_pretraining(
            _data=_data,
            test_dir=run_dir / f"test/ep_{epoch}",
            model=model,
            save_val=save_val and img_cnt <= config.save_max_idx,
            test_state=test_state,
            img_cnt=img_cnt,
        )

        img_cnt += batch_cnt

    log_summary(init_time=config.init_time, state=test_state, log_std=True)

    primary_metric = test_state.mean("psnr")
    return primary_metric
