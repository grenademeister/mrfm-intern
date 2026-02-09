import os
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import torch
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
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

        config.init_time = time.time()
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dir setting
        if config.resume and not config.resume_path:
            raise ValueError("resume is True but resume_path is empty.")
        self.resume_ckpt_path = None
        if config.resume:
            self.resume_ckpt_path = self._resolve_resume_ckpt_path(Path(config.resume_path))
            self.run_dir = self._resolve_run_dir(self.resume_ckpt_path)
        else:
            self.run_dir = config.run_dir / f"{call_next_id(config.run_dir):05d}_train"
        os.makedirs(self.run_dir, exist_ok=True)
        logger_add_handler(logger, f"{self.run_dir/'log.log'}", config.log_lv)
        logger.info(separator())
        logger.info(f"Run dir: {self.run_dir}")
        if self.resume_ckpt_path is not None:
            logger.info(f"Resume checkpoint: {self.resume_ckpt_path}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.tb_writer = None
        if config.tb_enable:
            tb_dir = self.run_dir / config.tb_log_dir
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))

        # log config
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
                )
                state_dict = self._normalize_state_dict(self.resume_state["model_state_dict"], is_parallel=False)
                self.network.load_state_dict(state_dict, strict=True)
            else:
                self.network = load_from_ckpt(
                    ckpt_path=Path(config.pretrained),
                    from_scratch=config.from_scratch,
                    use_vision_decoder=True,
                    use_vision_decoder_weights=False,
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
        )

        self.train_loader, _, self.train_len = get_data_wrapper_loader(
            file_path=config.train_dataset,
            training_mode=True,
            loader_cfg=train_loader_cfg,
            split="train",
        )
        logger.info(f"Train dataset length : {self.train_len}")

        self.valid_loader, _, valid_len = get_data_wrapper_loader(
            file_path=config.valid_dataset,
            training_mode=False,
            loader_cfg=valid_loader_cfg,
            split="valid",
        )
        logger.info(f"Valid dataset length : {valid_len}")

        self.test_loader, _, test_len = get_data_wrapper_loader(
            file_path=config.test_dataset,
            training_mode=False,
            loader_cfg=test_loader_cfg,
            split="test",
        )
        logger.info(f"Test dataset length : {test_len}")

    @error_wrap
    def _set_network(
        self,
    ) -> None:
        if config.parallel:
            self.network = torch.nn.DataParallel(self.network).to(config.device)
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
        logger.info(separator())
        logger.info("Train start")

        best_metric: float = 0

        start_epoch = self.start_epoch
        if config.resume and start_epoch > 0:
            logger.info(f"Resume start epoch: {start_epoch}")
        for epoch in range(start_epoch, config.train_epoch):
            logger.info(f"Epoch: {epoch}")
            lr_epoch = get_learning_rate(
                epoch=epoch,
                lr=config.lr,
                lr_decay=config.lr_decay,
                lr_tol=config.lr_tol,
            )

            optims = [set_optimizer_lr(optimizer=optim, learning_rate=lr_epoch) for optim in self.optims]
            logger.info(f"Learning rate: {lr_epoch:0.3e}")
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("train/lr", lr_epoch, epoch)

            train_epoch(
                train_loader=self.train_loader,
                train_len=self.train_len,
                network=self.network,
                optim_list=optims,
                epoch=epoch,
                tb_writer=self.tb_writer,
                tb_log_batch=config.tb_log_batch,
            )

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

            if primary_metric is not None and primary_metric > best_metric:
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
        logger.info("Valid")
        with torch.no_grad():
            primary_metric = test_part(
                epoch=epoch,
                data_loader=self.valid_loader,
                network=self.network,
                run_dir=self.run_dir,
                save_val=False,
                tb_writer=self.tb_writer,
                tb_prefix="valid",
            )
        return primary_metric

    @error_wrap
    def _test(
        self,
        epoch: int,
    ) -> None:
        logger.info("Test")
        with torch.no_grad():
            test_part(
                epoch=epoch,
                data_loader=self.test_loader,
                network=self.network,
                run_dir=self.run_dir,
                save_val=config.save_val,
                tb_writer=self.tb_writer,
                tb_prefix="test",
            )


if __name__ == "__main__":
    trainer = Trainer()
    trainer()
