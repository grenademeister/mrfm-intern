import os
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common.logger import logger, logger_add_handler
from common.utils import (
    call_next_id,
    separator,
)
from common.wrapper import error_wrap
from core_funcs import NETWORK, OPTIM, ModelType, get_learning_rate, get_optim, save_checkpoint, set_optimizer_lr, test_part, train_epoch
from datawrapper.datawrapper import LoaderConfig, get_data_wrapper_loader
from model.listfm_it import load_from_ckpt
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

    def __init__(
        self,
    ) -> None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

        config.init_time = time.time()
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dir setting
        self.run_dir = config.run_dir / f"{call_next_id(config.run_dir):05d}_train"
        logger_add_handler(logger, f"{self.run_dir/'log.log'}", config.log_lv)
        logger.info(separator())
        logger.info(f"Run dir: {self.run_dir}")
        os.makedirs(self.run_dir, exist_ok=True)

        # log config
        logger.info(separator())
        logger.info("General Config")
        config_dict = asdict(config)
        for k in config_dict:
            logger.info(f"{k}:{config_dict[k]}")

    def __call__(
        self,
    ) -> None:
        self._set_data()
        self._set_network()
        self._train()

    @error_wrap
    def _set_data(
        self,
    ) -> None:
        if ModelType.from_string(config.model_type) == ModelType.LISTFM_IT:
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

        self.optims = [
            get_optim(
                network=self.network,
                optimizer=config.optimizer,
            ),
        ]

        if config.parallel:
            self.network = torch.nn.DataParallel(self.network).to(config.device)
        else:
            self.network = self.network.to(config.device)

    @error_wrap
    def _train(
        self,
    ) -> None:
        logger.info(separator())
        logger.info("Train start")

        best_metric: float = 0

        for epoch in range(config.train_epoch):
            logger.info(f"Epoch: {epoch}")
            lr_epoch = get_learning_rate(
                epoch=epoch,
                lr=config.lr,
                lr_decay=config.lr_decay,
                lr_tol=config.lr_tol,
            )

            optims = [set_optimizer_lr(optimizer=optim, learning_rate=lr_epoch) for optim in self.optims]
            logger.info(f"Learning rate: {lr_epoch:0.3e}")

            train_epoch(
                train_loader=self.train_loader,
                train_len=self.train_len,
                network=self.network,
                optim_list=optims,
                epoch=epoch,
            )

            save_checkpoint(
                network=self.network,
                run_dir=self.run_dir,
                epoch=epoch,
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
            )


if __name__ == "__main__":
    trainer = Trainer()
    trainer()
