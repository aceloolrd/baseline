from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from config_loader import train_config

callbacks = [
    ModelCheckpoint(
        dirpath=f"{train_config['DEFAULT_ROOT_DIR']}/checkpoints/{train_config['EXPERIMENT']}",
        filename="{epoch:02d}_{valid_dataset_iou:.4f}_{valid_loss:.4f}",
        save_top_k=3,
        monitor="valid_loss",
        mode="min",
        save_last=True,
        every_n_epochs=1,
    ),

    EarlyStopping(
        monitor="valid_loss",
        min_delta=1e-3,
        patience=5,
        mode="min",
    ),

    LearningRateMonitor(logging_interval="epoch"),
]
