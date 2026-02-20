import math
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING
from dataclasses import asdict
from enum import Enum
from pathlib import Path

import torch
import torch.distributed as dist
from scipy.io import savemat
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
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

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

NETWORK = LISTFoundationModelIT | torch.nn.DataParallel[LISTFoundationModelIT] | torch.nn.parallel.DistributedDataParallel
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
    x0 = torch.randn_like(label) * config.flow_noise_std  # noise
    xt = (1.0 - t) * x0 + t * label  # noisy z
    v = label - x0  # velocity
    return v, xt


def _rectified_flow_sample_dep(
    model: NETWORK,
    img: Tensor,
    text: Tensor,
    instruction: Tensor,
    instruction_llm_ids: Tensor | None = None,
    instruction_llm_mask: Tensor | None = None,
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
            instruction_llm_ids=instruction_llm_ids,
            instruction_llm_mask=instruction_llm_mask,
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
        instruction_llm_ids=instruction_llm_ids,
        instruction_llm_mask=instruction_llm_mask,
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
    instruction_llm_ids: Tensor | None = None,
    instruction_llm_mask: Tensor | None = None,
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
            instruction_llm_ids=instruction_llm_ids,
            instruction_llm_mask=instruction_llm_mask,
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
    tb_writer: "SummaryWriter | None" = None,
    tb_prefix: str | None = None,
    step: int | None = None,
) -> None:
    spend_time = seconds_to_dhms(time.time() - init_time)
    for key in state.state_dict:
        if log_std:
            summary = f"{spend_time} | {key}: {state.mean(key):0.3e} + {state.std(key):0.3e} "
            logger.info(summary)
            if tb_writer is not None and tb_prefix is not None and step is not None:
                tb_writer.add_scalar(f"{tb_prefix}/{key}", state.mean(key), step)
                tb_writer.add_scalar(f"{tb_prefix}/{key}_std", state.std(key), step)
        else:
            summary = f"{spend_time} | {key}: {state.mean(key):0.3e}"
            logger.info(summary)
            if tb_writer is not None and tb_prefix is not None and step is not None:
                tb_writer.add_scalar(f"{tb_prefix}/{key}", state.mean(key), step)


def save_checkpoint(
    network: NETWORK,
    run_dir: Path,
    epoch: str | int | None = None,
    optims: list[OPTIM | None] | None = None,
    epoch_idx: int | None = None,
) -> None:
    if epoch is None:
        epoch = "best"
    os.makedirs(run_dir / "checkpoints", exist_ok=True)
    optim_state_dicts = None
    if optims is not None:
        optim_state_dicts = [optim.state_dict() if optim is not None else None for optim in optims]
    stored_epoch = epoch_idx
    if stored_epoch is None and isinstance(epoch, int):
        stored_epoch = epoch
    if ModelType.from_string(config.model_type) == ModelType.LISTFM_IT:
        torch.save(
            {
                "model_state_dict": network.state_dict(),
                "model_config": asdict(
                    network.module.listfmconfig
                    if isinstance(
                        network,
                        (
                            torch.nn.DataParallel,
                            torch.nn.parallel.DistributedDataParallel,
                        ),
                    )
                    else network.listfmconfig
                ),
                "epoch": stored_epoch,
                "optim_state_dicts": optim_state_dicts,
            },
            run_dir / f"checkpoints/checkpoint_{epoch}.ckpt",
        )
    elif ModelType.from_string(config.model_type) == ModelType.UNET:
        torch.save(
            {
                "model_state_dict": network.state_dict(),
                "epoch": stored_epoch,
                "optim_state_dicts": optim_state_dicts,
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
    tesner_dict: dict[str, Tensor | str | tuple[str, ...] | list[str] | None],
    img_cnt: int,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    os.makedirs(test_dir, exist_ok=True)
    save_dict = {}

    if batch_cnt == 0:
        logger.warning("batch_cnt is 0, no data to save")
        return

    for i in range(batch_cnt):
        for key, value in tesner_dict.items():
            if value is None:
                continue
            if torch.is_tensor(value):
                save_dict[key] = value.cpu().detach().numpy()[i, ...]
            elif isinstance(value, (list, tuple)):
                save_dict[key] = value[i]
            else:
                save_dict[key] = value

        # Global unique index across ranks: interleave by rank
        idx = (img_cnt + i) * world_size + rank + 1
        savemat(f"{test_dir}/{idx}_res.mat", save_dict)


def train_epoch_listfm_vision_pretraining(
    _data: dict[DataKey, Tensor | str],
    network: NETWORK,
    epoch: int,
    train_state: MetricController,
) -> tuple[int, float]:
    loss_func = get_loss_func(config.loss_model)

    input: Tensor = _data[DataKey.Input].to(config.device)
    text: Tensor = _data[DataKey.Text].to(config.device)
    label: Tensor = _data[DataKey.Label].to(config.device)
    instruction: Tensor = _data[DataKey.Instruction].to(config.device)
    instruction_raw: tuple[str, ...] = _data[DataKey.InstructionRaw]
    if config.text_encoding == "clip":
        instruction_llm_ids = None
        instruction_llm_mask = _data[DataKey.InstructionLLMAttention].to(config.device)
    else:
        instruction_llm_ids = _data[DataKey.InstructionLLMIds].to(config.device)
        instruction_llm_mask = _data[DataKey.InstructionLLMAttention].to(config.device)
    img_cnt_minibatch = input.shape[0]

    flow_t = sample_flow_t(batch=img_cnt_minibatch, device=config.device)
    flow_v, flow_xt = make_flow_pair(label=label, t=flow_t)

    output = network.forward(
        img=input,
        text=text,
        instruction=instruction,
        instruction_llm_ids=instruction_llm_ids,
        instruction_llm_mask=instruction_llm_mask,
        use_bottleneck=config.use_bottleneck,
        grad_encoder=config.grad_encoder,
        flow_xt=flow_xt,
        flow_t=flow_t.view(img_cnt_minibatch, 1),
    )
    # pred_v = (output - flow_xt) / (1.0 - flow_t).clamp_min(config.flow_eval_eps)
    loss = torch.mean(loss_func(output, label), dim=(1, 2, 3), keepdim=True)

    loss_mean = torch.mean(loss)
    loss_mean.backward()
    train_state.add("loss", loss)

    return img_cnt_minibatch, float(loss_mean.detach().cpu().item())


def train_epoch(
    train_loader: DataLoader,
    train_len: int,
    network: NETWORK,
    optim_list: list[OPTIM | None],
    epoch: int,
    tb_writer: "SummaryWriter | None" = None,
    tb_log_batch: bool = False,
    log_enabled: bool = True,
) -> None:
    train_state = MetricController()
    train_state.reset()
    network.train()

    logging_cnt: int = 1
    img_cnt: int = 0
    total_batches = len(train_loader)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    log_interval = max(1, int(train_len / (config.logging_density * max(1, world_size))))
    for batch_idx, _data in enumerate(train_loader):
        zero_optimizers(optim_list=optim_list)
        img_cnt_minibatch, loss_mean = train_epoch_listfm_vision_pretraining(
            _data=_data,
            network=network,
            epoch=epoch,
            train_state=train_state,
        )

        step_optimizers(optim_list=optim_list)
        if tb_writer is not None and tb_log_batch and log_enabled:
            step = epoch * total_batches + batch_idx
            tb_writer.add_scalar("train/loss_batch", loss_mean, step)
        img_cnt += img_cnt_minibatch
        if log_enabled and img_cnt > (log_interval * logging_cnt):
            log_summary(
                init_time=config.init_time,
                state=train_state,
                tb_writer=tb_writer,
                tb_prefix="train",
                step=epoch,
            )
            logging_cnt += 1

    if log_enabled:
        log_summary(
            init_time=config.init_time,
            state=train_state,
            tb_writer=tb_writer,
            tb_prefix="train",
            step=epoch,
        )


def test_part_listfm_vision_pretraining(
    _data: dict[DataKey, Tensor | str],
    test_dir: Path,
    model: NETWORK,
    save_val: bool,
    test_state: MetricController,
    img_cnt: int,
    rank: int = 0,
    world_size: int = 1,
    task_states: dict[str, MetricController] | None = None,
) -> float:
    loss_func = get_loss_func(config.loss_model)

    input: Tensor = _data[DataKey.Input].to(config.device)
    text: Tensor = _data[DataKey.Text].to(config.device)
    label: Tensor = _data[DataKey.Label].to(config.device)
    instruction: Tensor = _data[DataKey.Instruction].to(config.device)
    instruction_raw: tuple[str, ...] = _data[DataKey.InstructionRaw]
    if config.text_encoding == "clip":
        instruction_llm_ids = None
        instruction_llm_mask = _data[DataKey.InstructionLLMAttention].to(config.device)
    else:
        instruction_llm_ids = _data[DataKey.InstructionLLMIds].to(config.device)
        instruction_llm_mask = _data[DataKey.InstructionLLMAttention].to(config.device)
    task_names: tuple[str, ...] = _data[DataKey.TaskName]

    batch_cnt = input.shape[0]
    with torch.no_grad():
        output = rectified_flow_sample(
            model=model,
            img=input,
            text=text,
            instruction=instruction,
            instruction_llm_ids=instruction_llm_ids,
            instruction_llm_mask=instruction_llm_mask,
        )

    loss = torch.mean(loss_func(output, label), dim=(1, 2, 3), keepdim=True)

    # Convert to numpy once for all metrics
    output_np = output.detach().cpu().numpy()
    label_np = label.detach().cpu().numpy()
    psnr_list: list[float] = []
    ssim_list: list[float] = []
    dice_list: list[float] = []
    # Crop region for metric computation (matches visualization)
    x1, x2, y1, y2 = 20, -20, 20, -20
    for i in range(batch_cnt):
        out_i = output_np[i]
        lab_i = label_np[i]
        task_name = task_names[i]
        is_seg = "segmentation" in task_name
        if is_seg:
            if out_i.ndim == 3 and out_i.shape[0] == 1:
                out_i = out_i[0]
                lab_i = lab_i[0]
            pred_bin = out_i > 0.5
            lab_bin = lab_i > 0.5
            intersection = (pred_bin & lab_bin).sum()
            denom = pred_bin.sum() + lab_bin.sum()
            dice = (2.0 * intersection + 1e-7) / (denom + 1e-7)
            dice_list.append(float(dice))
            continue
        channel_axis = None
        if out_i.ndim == 3 and out_i.shape[0] == 1:
            out_i = out_i[0]
            lab_i = lab_i[0]
        elif out_i.ndim == 3:
            channel_axis = 0
        # Apply crop on spatial dims for non-seg metrics
        if out_i.ndim == 2:
            out_i = out_i[x1:x2, y1:y2]
            lab_i = lab_i[x1:x2, y1:y2]
        elif out_i.ndim == 3 and channel_axis == 0:
            out_i = out_i[:, x1:x2, y1:y2]
            lab_i = lab_i[:, x1:x2, y1:y2]
        data_range = float(lab_i.max() - lab_i.min())
        psnr_list.append(peak_signal_noise_ratio(out_i, lab_i, data_range=data_range))
        ssim_list.append(
            structural_similarity(out_i, lab_i, data_range=data_range, channel_axis=channel_axis)
        )
    psnr = torch.tensor(psnr_list) if psnr_list else None
    ssim = torch.tensor(ssim_list) if ssim_list else None
    dice = torch.tensor(dice_list) if dice_list else None

    # Add to overall metrics
    test_state.add("loss", loss)
    if psnr is not None:
        test_state.add("psnr", psnr)
    if ssim is not None:
        test_state.add("ssim", ssim)
    if dice is not None:
        test_state.add("dice", dice)

    # Add to task-specific metrics (per sample in batch)
    if task_states is not None:
        seg_idx = 0
        nonseg_idx = 0
        for i in range(batch_cnt):
            task_name = task_names[i]
            if task_name not in task_states:
                task_states[task_name] = MetricController()
                task_states[task_name].reset()
            
            # Add metrics for individual sample
            task_states[task_name].add("loss", loss[i:i+1])
            is_seg = "segmentation" in task_name
            if is_seg:
                if dice is not None:
                    task_states[task_name].add("dice", dice[seg_idx:seg_idx + 1])
                    seg_idx += 1
            else:
                if psnr is not None:
                    task_states[task_name].add("psnr", psnr[nonseg_idx:nonseg_idx + 1])
                if ssim is not None:
                    task_states[task_name].add("ssim", ssim[nonseg_idx:nonseg_idx + 1])
                nonseg_idx += 1
    
    # Explicitly delete tensors to free memory
    del loss, psnr, ssim, dice, output_np, label_np, psnr_list, ssim_list, dice_list

    if save_val:
        save_result_to_mat(
            test_dir=test_dir,
            batch_cnt=batch_cnt,
            tesner_dict={
                "input": input,
                "out": output,
                "label": label,
                "text": text,
                "instruction": instruction,
                "instruction_raw": instruction_raw,
                "task_name": task_names,
            },
            img_cnt=img_cnt,
            rank=rank,
            world_size=world_size,
        )
        # Clear GPU cache after saving
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return batch_cnt


def test_part(
    epoch: int,
    data_loader: DataLoader,
    network: NETWORK,
    run_dir: Path,
    save_val: bool,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    log_enabled: bool = True,
    tb_writer: "SummaryWriter | None" = None,
    tb_prefix: str = "valid",
) -> float:
    test_state = MetricController()
    test_state.reset()
    task_states: dict[str, MetricController] = {}
    network.eval()
    model = network

    img_cnt: int = 0
    for _data in data_loader:
        global_img_cnt = img_cnt * world_size + rank
        batch_cnt = test_part_listfm_vision_pretraining(
            _data=_data,
            test_dir=run_dir / f"test/ep_{epoch}",
            model=model,
            save_val=save_val and global_img_cnt <= config.save_max_idx,
            test_state=test_state,
            img_cnt=img_cnt,
            rank=rank,
            world_size=world_size,
            task_states=task_states,
        )

        img_cnt += batch_cnt

    if distributed and dist.is_available() and dist.is_initialized():
        global_stats: dict[str, tuple[float, float]] = {}
        for key, values in test_state.state_dict.items():
            if not values:
                continue
            local_sum = float(sum(values))
            local_sumsq = float(sum(v * v for v in values))
            local_count = float(len(values))
            stats = torch.tensor([local_sum, local_sumsq, local_count], device=config.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_sum, total_sumsq, total_count = stats.tolist()
            if total_count <= 0:
                mean = 0.0
                std = 0.0
            else:
                mean = total_sum / total_count
                if total_count > 1:
                    var = (total_sumsq - (total_sum * total_sum) / total_count) / (total_count - 1)
                    std = math.sqrt(max(var, 0.0))
                else:
                    std = 0.0
            global_stats[key] = (mean, std)

        # Task-specific metrics (gathered across ranks)
        merged_task_names: list[str] = []
        if task_states:
            local_task_names = sorted(task_states.keys())
            all_task_names: list[list[str]] = [None] * world_size
            dist.all_gather_object(all_task_names, local_task_names)
            merged_task_names = sorted({name for sub in all_task_names for name in sub})

        task_global_stats: dict[tuple[str, str], tuple[float, float]] = {}
        for task_name in merged_task_names:
            keys = ("loss", "dice") if "segmentation" in task_name else ("loss", "psnr", "ssim")
            for key in keys:
                values = task_states.get(task_name, MetricController()).state_dict.get(key, [])
                local_sum = float(sum(values)) if values else 0.0
                local_sumsq = float(sum(v * v for v in values)) if values else 0.0
                local_count = float(len(values)) if values else 0.0
                stats = torch.tensor([local_sum, local_sumsq, local_count], device=config.device)
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                total_sum, total_sumsq, total_count = stats.tolist()
                if total_count <= 0:
                    mean = 0.0
                    std = 0.0
                else:
                    mean = total_sum / total_count
                    if total_count > 1:
                        var = (total_sumsq - (total_sum * total_sum) / total_count) / (total_count - 1)
                        std = math.sqrt(max(var, 0.0))
                    else:
                        std = 0.0
                task_global_stats[(task_name, key)] = (mean, std)

        if log_enabled:
            logger.info("=" * 80)
            logger.info("Overall Metrics:")
            spend_time = seconds_to_dhms(time.time() - config.init_time)
            for key, (mean, std) in global_stats.items():
                logger.info(f"{spend_time} | {key}: {mean:0.3e} + {std:0.3e}")
                if tb_writer is not None and tb_prefix is not None and epoch is not None:
                    tb_writer.add_scalar(f"{tb_prefix}/{key}", mean, epoch)
                    tb_writer.add_scalar(f"{tb_prefix}/{key}_std", std, epoch)
            logger.info("=" * 80)

            if merged_task_names:
                logger.info("Task-specific Metrics:")
                for task_name in merged_task_names:
                    logger.info(f"--- {task_name} ---")
                    keys = ("loss", "dice") if "segmentation" in task_name else ("loss", "psnr", "ssim")
                    for key in keys:
                        mean, std = task_global_stats[(task_name, key)]
                        logger.info(f"{spend_time} | {key}: {mean:0.3e} + {std:0.3e}")
                        if tb_writer is not None and tb_prefix is not None and epoch is not None:
                            tb_writer.add_scalar(f"{tb_prefix}/{task_name}/{key}", mean, epoch)
                            tb_writer.add_scalar(f"{tb_prefix}/{task_name}/{key}_std", std, epoch)
                logger.info("=" * 80)

        primary_metric = global_stats.get("psnr", (0.0, 0.0))[0]
        if primary_metric == 0.0 and "dice" in global_stats:
            primary_metric = global_stats.get("dice", (0.0, 0.0))[0]
        return float(primary_metric)

    # Non-distributed logging
    if log_enabled:
        logger.info("=" * 80)
        logger.info("Overall Metrics:")
        log_summary(
            init_time=config.init_time,
            state=test_state,
            log_std=True,
            tb_writer=tb_writer,
            tb_prefix=tb_prefix,
            step=epoch,
        )

        # Log task-specific metrics
        if task_states:
            logger.info("=" * 80)
            logger.info("Task-specific Metrics:")
            for task_name in sorted(task_states.keys()):
                logger.info(f"--- {task_name} ---")
                log_summary(
                    init_time=config.init_time,
                    state=task_states[task_name],
                    log_std=True,
                    tb_writer=tb_writer,
                    tb_prefix=f"{tb_prefix}/{task_name}",
                    step=epoch,
                )
            logger.info("=" * 80)

    if "psnr" in test_state.state_dict:
        primary_metric = test_state.mean("psnr")
    else:
        primary_metric = test_state.mean("dice")
    
    # Clean up memory after validation/test
    test_state.reset()
    task_states.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return primary_metric
