import os
import sys

# [중요] LOCAL_RANK를 확인하여 import torch 전에 GPU를 물리적으로 격리합니다.
# 이렇게 하면 각 프로세스는 자기에게 할당된 GPU 1개만 '0번'으로 인식하게 됩니다.
if "LOCAL_RANK" in os.environ and "CUDA_VISIBLE_DEVICES" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(visible_devices) > local_rank:
        # 현재 프로세스가 사용할 물리적 GPU 번호 하나만 남깁니다.
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices[local_rank]

import time

import warnings
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common.logger import logger, logger_add_handler
from common.utils import (
    call_next_id,
    separator,
)
from common.wrapper import error_wrap
from core_funcs import NETWORK, OPTIM, ModelType, get_learning_rate, get_optim, save_checkpoint, set_optimizer_lr, test_part, train_epoch
from datawrapper.datawrapper import LoaderConfig, get_data_wrapper_loader
from model.listfm_backbone import LISTFMConfig
from model.listfm_it import LISTFoundationModelIT, load_from_ckpt
from model.unet import Unet
from params import config

warnings.filterwarnings("ignore")


class Trainer:
    run_dir: Path
    network: NETWORK
    train_loader: DataLoader
    train_len: int
    valid_loader: DataLoader
    optims: list[OPTIM | None]
    tb_writer: SummaryWriter | None
    resume_ckpt_path: Path | None
    resume_state: dict | None
    start_epoch: int

    def __init__(
        self,
    ) -> None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

        config.init_time = time.time()

        self.is_distributed = False
        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1
        if dist.is_available():
            world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
            if world_size_env > 1:
                self.is_distributed = True
                # 이미 격리되었으므로 여기서 local_rank는 항상 0이 됩니다.
                self.local_rank = 0
                self.global_rank = int(os.environ.get("RANK", "0"))
                self.world_size = world_size_env
                torch.cuda.set_device(0)
                dist.init_process_group(backend="nccl", init_method="env://")

        self.is_main_process = self.global_rank == 0
        if self.is_distributed:
            config.device = torch.device("cuda", 0)
        else:
            config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dir setting
        if config.resume and not config.resume_path:
            raise ValueError("resume is True but resume_path is empty.")
        self.resume_ckpt_path = None
        if config.resume:
            self.resume_ckpt_path = self._resolve_resume_ckpt_path(Path(config.resume_path))
            self.run_dir = self._resolve_run_dir(self.resume_ckpt_path)
        else:
            if self.is_distributed:
                if self.is_main_process:
                    run_dir = config.run_dir / f"{call_next_id(config.run_dir):05d}_train"
                    obj_list = [str(run_dir)]
                else:
                    obj_list = [""]
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast_object_list(obj_list, src=0)
                self.run_dir = Path(obj_list[0])
            else:
                self.run_dir = config.run_dir / f"{call_next_id(config.run_dir):05d}_train"
        os.makedirs(self.run_dir, exist_ok=True)
        if self.is_main_process:
            logger_add_handler(logger, f"{self.run_dir/'log.log'}", config.log_lv)
            logger.info(separator())
            logger.info(f"Run dir: {self.run_dir}")
            if self.resume_ckpt_path is not None:
                logger.info(f"Resume checkpoint: {self.resume_ckpt_path}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.tb_writer = None
        if config.tb_enable and self.is_main_process:
            tb_dir = self.run_dir / config.tb_log_dir
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))

        # log config
        if self.is_main_process:
            logger.info(separator())
            logger.info("General Config")
            config_dict = asdict(config)
            for k in config_dict:
                logger.info(f"{k}:{config_dict[k]}")

        self.resume_state = None
        self.start_epoch = 0

    def _resolve_run_dir(
        self,
        ckpt_path: Path,
    ) -> Path:
        if ckpt_path.parent.name == "checkpoints":
            return ckpt_path.parent.parent
        return ckpt_path.parent

    def _resolve_resume_ckpt_path(
        self,
        resume_path: Path,
    ) -> Path:
        if resume_path.is_file():
            return resume_path
        if not resume_path.is_dir():
            raise FileNotFoundError(f"Resume path {resume_path} does not exist.")
        ckpt_dir = resume_path / "checkpoints" if (resume_path / "checkpoints").is_dir() else resume_path
        candidates = list(ckpt_dir.glob("checkpoint_*.ckpt"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

        def _epoch_from_name(path: Path) -> int | None:
            stem = path.stem
            if "_" not in stem:
                return None
            suffix = stem.split("_", 1)[1]
            return int(suffix) if suffix.isdigit() else None

        numeric = [(epoch, path) for path in candidates if (epoch := _epoch_from_name(path)) is not None]
        if numeric:
            numeric.sort(key=lambda x: x[0])
            return numeric[-1][1]
        best = [path for path in candidates if path.stem.endswith("best")]
        if best:
            return best[0]
        candidates.sort(key=lambda p: p.stat().st_mtime)
        return candidates[-1]

    def _normalize_state_dict(
        self,
        state_dict: dict,
        is_parallel: bool,
    ) -> dict:
        has_module = any(key.startswith("module.") for key in state_dict)
        if has_module and not is_parallel:
            return {key[7:]: value for key, value in state_dict.items()}
        if not has_module and is_parallel:
            return {f"module.{key}": value for key, value in state_dict.items()}
        return state_dict

    def __call__(
        self,
    ) -> None:
        try:
            self._set_data()
            self._set_network()
            self._train()
        finally:
            if self.tb_writer is not None:
                self.tb_writer.flush()
                self.tb_writer.close()
            if self.is_distributed and dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()

    @error_wrap
    def _set_data(
        self,
    ) -> None:
        if config.resume:
            if self.resume_ckpt_path is None:
                raise ValueError("resume is True but resume_ckpt_path is not set.")
            self.resume_state = torch.load(
                self.resume_ckpt_path,
                map_location="cpu",
                weights_only=False,
            )
            stored_epoch = self.resume_state.get("epoch")
            if stored_epoch is not None:
                self.start_epoch = int(stored_epoch) + 1
        if ModelType.from_string(config.model_type) == ModelType.LISTFM_IT:
            if config.resume:
                if self.resume_state is None or "model_config" not in self.resume_state:
                    raise KeyError("Resume checkpoint missing model_config.")
                listfmconfig = LISTFMConfig(**self.resume_state["model_config"])
                listfmconfig.text_enc_pretrained = None
                self.network = LISTFoundationModelIT(
                    listfmconfig,
                    use_vision_decoder=True,
                    qwen_model_path=config.qwen_model_path,
                    qwen_lora_path=config.qwen_lora_path,
                    qwen_trainable=config.qwen_trainable,
                )
                state_dict = self._normalize_state_dict(self.resume_state["model_state_dict"], is_parallel=False)
                self.network.load_state_dict(state_dict, strict=True)
            else:
                self.network = load_from_ckpt(
                    ckpt_path=Path(config.pretrained),
                    from_scratch=config.from_scratch,
                    use_vision_decoder=True,
                    use_vision_decoder_weights=False,
                    qwen_model_path=config.qwen_model_path,
                    qwen_lora_path=config.qwen_lora_path,
                    qwen_trainable=config.qwen_trainable,
                )
            logger.info(separator())
            logger.info("Model Config")
            config_dict = asdict(self.network.listfmconfig)
            for k in config_dict:
                logger.info(f"{k}:{config_dict[k]}")
        elif ModelType.from_string(config.model_type) == ModelType.UNET:
            self.network = Unet(
                in_chans=1,
                out_chans=1,
                chans=64,
                num_pool_layers=4,
            )
            if config.resume:
                if self.resume_state is None or "model_state_dict" not in self.resume_state:
                    raise KeyError("Resume checkpoint missing model_state_dict.")
                state_dict = self._normalize_state_dict(self.resume_state["model_state_dict"], is_parallel=False)
                self.network.load_state_dict(state_dict, strict=True)
            logger.info(separator())
            logger.info("Model Config")
            config_dict = {
                "in_chans": 1,
                "out_chans": 1,
                "chans": 64,
                "num_pool_layers": 4,
            }
            for k in config_dict:
                logger.info(f"{k}:{config_dict[k]}")
        else:
            raise KeyError("model type not matched")

        logger.info(separator())
        train_loader_cfg = LoaderConfig(
            batch=config.train_batch,
            num_workers=config.num_workers,
            shuffle=True,
            debug_mode=config.debugmode,
            acs_num=config.acs_num,
            parallel_factor=config.parallel_factor,
            data_type=config.data_type,
            subject_num=config.subject_num,
            train_percent=config.train_percent,
            slice_per_subject=config.slice_per_subject,
            qwen_model_path=config.qwen_model_path,
            qwen_max_length=config.qwen_max_length,
            qwen_use_fast=config.qwen_use_fast,
        )

        valid_loader_cfg = LoaderConfig(
            batch=config.valid_batch,
            num_workers=config.num_workers,
            shuffle=True,
            debug_mode=config.debugmode,
            acs_num=config.acs_num,
            parallel_factor=config.parallel_factor,
            data_type=config.data_type,
            subject_num=config.subject_num,
            train_percent=config.train_percent,
            slice_per_subject=config.slice_per_subject,
            qwen_model_path=config.qwen_model_path,
            qwen_max_length=config.qwen_max_length,
            qwen_use_fast=config.qwen_use_fast,
        )

        test_loader_cfg = LoaderConfig(
            batch=config.valid_batch,
            num_workers=config.num_workers,
            shuffle=True,
            debug_mode=config.debugmode,
            acs_num=config.acs_num,
            parallel_factor=config.parallel_factor,
            data_type=config.data_type,
            subject_num=config.subject_num,
            train_percent=config.train_percent,
            slice_per_subject=config.slice_per_subject,
            qwen_model_path=config.qwen_model_path,
            qwen_max_length=config.qwen_max_length,
            qwen_use_fast=config.qwen_use_fast,
        )

        self.train_loader, _, self.train_len = get_data_wrapper_loader(
            file_path=config.train_dataset,
            training_mode=True,
            loader_cfg=train_loader_cfg,
            split="train",
            distributed=self.is_distributed,
            rank=self.global_rank,
            world_size=self.world_size,
        )
        logger.info(f"Train dataset length : {self.train_len}")

        self.valid_loader, _, valid_len = get_data_wrapper_loader(
            file_path=config.valid_dataset,
            training_mode=False,
            loader_cfg=valid_loader_cfg,
            split="valid",
            distributed=False,
            rank=0,
            world_size=1,
        )
        logger.info(f"Valid dataset length : {valid_len}")

        self.test_loader, _, test_len = get_data_wrapper_loader(
            file_path=config.test_dataset,
            training_mode=False,
            loader_cfg=test_loader_cfg,
            split="test",
            distributed=False,
            rank=0,
            world_size=1,
        )
        logger.info(f"Test dataset length : {test_len}")

    @error_wrap
    def _set_network(
        self,
    ) -> None:
        if self.is_distributed:
            self.network = torch.nn.parallel.DistributedDataParallel(
                self.network.to(config.device),
                device_ids=[0],
                output_device=0,
                find_unused_parameters=True,
            )
        else:
            self.network = self.network.to(config.device)

        self.optims = [
            get_optim(
                network=self.network,
                optimizer=config.optimizer,
            ),
        ]
        if config.resume and self.resume_state is not None:
            optim_state_dicts = self.resume_state.get("optim_state_dicts")
            if optim_state_dicts is not None:
                for optim, state in zip(self.optims, optim_state_dicts):
                    if optim is not None and state is not None:
                        optim.load_state_dict(state)
                        for value in optim.state.values():
                            for key, tensor in value.items():
                                if torch.is_tensor(tensor):
                                    value[key] = tensor.to(config.device)

    @error_wrap
    def _train(
        self,
    ) -> None:
        if self.is_main_process:
            logger.info(separator())
            logger.info("Train start")

        best_metric: float = 0

        start_epoch = self.start_epoch
        if config.resume and start_epoch > 0:
            logger.info(f"Resume start epoch: {start_epoch}")
        for epoch in range(start_epoch, config.train_epoch):
            if self.is_main_process:
                logger.info(f"Epoch: {epoch}")
            lr_epoch = get_learning_rate(
                epoch=epoch,
                lr=config.lr,
                lr_decay=config.lr_decay,
                lr_tol=config.lr_tol,
            )

            optims = [set_optimizer_lr(optimizer=optim, learning_rate=lr_epoch) for optim in self.optims]
            if self.is_main_process:
                logger.info(f"Learning rate: {lr_epoch:0.3e}")
            if self.tb_writer is not None and self.is_main_process:
                self.tb_writer.add_scalar("train/lr", lr_epoch, epoch)

            if self.is_distributed and hasattr(self.train_loader, "sampler"):
                sampler = self.train_loader.sampler
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

            train_epoch(
                train_loader=self.train_loader,
                train_len=self.train_len,
                network=self.network,
                optim_list=optims,
                epoch=epoch,
                tb_writer=self.tb_writer,
                tb_log_batch=config.tb_log_batch,
                log_enabled=self.is_main_process,
            )

            if self.is_main_process:
                save_checkpoint(
                    network=self.network,
                    run_dir=self.run_dir,
                    epoch=epoch,
                    optims=optims,
                    epoch_idx=epoch,
                )

            if epoch < config.valid_tol:
                continue

            primary_metric = None
            if epoch % config.valid_interval == 0:
                primary_metric = self._valid(epoch)
                self._test(epoch)

            if self.is_main_process and primary_metric is not None and primary_metric > best_metric:
                best_metric = primary_metric
                logger.success("Best model renew")
                save_checkpoint(
                    network=self.network,
                    run_dir=self.run_dir,
                    optims=optims,
                    epoch_idx=epoch,
                )

    @error_wrap
    def _valid(
        self,
        epoch: int,
    ) -> float:
        if self.is_main_process:
            logger.info("Valid")
        with torch.no_grad():
            primary_metric = test_part(
                epoch=epoch,
                data_loader=self.valid_loader,
                network=self.network,
                run_dir=self.run_dir,
                save_val=False,
                distributed=self.is_distributed,
                rank=self.global_rank,
                world_size=self.world_size,
                log_enabled=self.is_main_process,
                tb_writer=self.tb_writer if self.is_main_process else None,
                tb_prefix="valid",
            )
        return primary_metric

    @error_wrap
    def _test(
        self,
        epoch: int,
    ) -> None:
        if self.is_main_process:
            logger.info("Test")
        with torch.no_grad():
            test_part(
                epoch=epoch,
                data_loader=self.test_loader,
                network=self.network,
                run_dir=self.run_dir,
                save_val=config.save_val,
                distributed=self.is_distributed,
                rank=self.global_rank,
                world_size=self.world_size,
                log_enabled=self.is_main_process,
                tb_writer=self.tb_writer if self.is_main_process else None,
                tb_prefix="test",
            )


if __name__ == "__main__":
    trainer = Trainer()
    trainer()
