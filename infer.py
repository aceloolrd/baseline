from pathlib import Path

import pytorch_lightning as pl

from config_loader import train_config
from datamodule import SegmentationDataModule
from loss import BCEDiceLoss
from model import SegmentationModel


if __name__ == "__main__":
    dm = SegmentationDataModule()
    dm.setup(stage="test")

    criterion = BCEDiceLoss()
    ckpt = (
        Path(train_config["DEFAULT_ROOT_DIR"])
        / "checkpoints"
        / train_config["EXPERIMENT"]
        / "last.ckpt"
    )

    model = SegmentationModel.load_from_checkpoint(
        str(ckpt),
        arch=train_config["ARCH"],
        encoder_name=train_config["ENCODER_NAME"],
        encoder_weights=train_config["ENCODER_WEIGHTS"],
        in_channels=train_config["IN_CHANNELS"],
        out_classes=train_config["OUT_CLASSES"],
        criterion=criterion,
        lr=train_config["LR"],
        batch_size=train_config["BATCH_SIZE"],
        epochs=train_config["EPOCHS"],
        steps_per_epoch=len(dm.test_dataloader()),
        mode=train_config.get("MODE", "binary"),
    )

    trainer = pl.Trainer(
        accelerator=train_config.get("ACCELERATOR", "gpu"),
        devices=1,
        logger=False,
        enable_model_summary=False,
    )

    trainer.test(model, dm)
