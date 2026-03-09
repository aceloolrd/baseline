import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from callbacks import callbacks
from config_loader import train_config
from datamodule import SegmentationDataModule
from loss import BCEDiceLoss
from model import SegmentationModel


if __name__ == "__main__":
    os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"

    seed_everything(train_config.get("SEED", 42), workers=True)

    dm = SegmentationDataModule()
    dm.setup(stage="fit")

    steps_per_epoch = len(dm.train_dataloader())
    criterion = BCEDiceLoss()

    model = SegmentationModel(
        arch=train_config.get("ARCH", "unetplusplus"),
        encoder_name=train_config.get("ENCODER_NAME", "resnext50_32x4d"),
        encoder_weights=train_config.get("ENCODER_WEIGHTS", "ssl"),
        in_channels=train_config.get("IN_CHANNELS", 1),
        out_classes=train_config.get("OUT_CLASSES", 1),
        criterion=criterion,
        lr=train_config.get("LR", 0.0001),
        batch_size=train_config.get("BATCH_SIZE", 4),
        epochs=train_config.get("EPOCHS", 50),
        steps_per_epoch=steps_per_epoch,
        mode=train_config.get("MODE", "binary"),
    )

    log_dir = Path(__file__).parent / train_config["DEFAULT_ROOT_DIR"] / "tb_logs"
    logger = TensorBoardLogger(str(log_dir), name=train_config["EXPERIMENT"])

    ckpt_path: str | None = train_config.get("CKPT_PATH")
    if isinstance(ckpt_path, str) and ckpt_path.lower() == "none":
        ckpt_path = None

    trainer = pl.Trainer(
        max_epochs=train_config.get("EPOCHS", 50),
        log_every_n_steps=1,
        callbacks=callbacks,
        accelerator=train_config.get("ACCELERATOR", "gpu"),
        devices=1,
        precision=train_config.get("PRECISION", "32"),
        enable_model_summary=True,
        enable_checkpointing=True,
        default_root_dir=train_config.get("DEFAULT_ROOT_DIR", "experiments"),
        logger=logger,
    )

    trainer.fit(model, dm, ckpt_path=ckpt_path)
