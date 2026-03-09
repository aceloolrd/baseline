from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from config_loader import train_config
from dataset import SegmentationDataset, train_transform, val_transform


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = train_config.get("DATASET_PATH", "./data/splits"),
        batch_size: int = train_config.get("BATCH_SIZE", 4),
        num_workers: int = train_config.get("NUM_WORKERS", 4),
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        if not Path(self.data_dir).exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.data_dir}")

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            self.train_dataset = SegmentationDataset(
                image_dir=f"{self.data_dir}/train/image",
                mask_dir=f"{self.data_dir}/train/mask",
                transform=train_transform,
            )
            self.val_dataset = SegmentationDataset(
                image_dir=f"{self.data_dir}/val/image",
                mask_dir=f"{self.data_dir}/val/mask",
                transform=val_transform,
            )

        if stage == "test":
            self.test_dataset = SegmentationDataset(
                image_dir=f"{self.data_dir}/test/image",
                mask_dir=f"{self.data_dir}/test/mask",
                transform=val_transform,
            )

    def _dataloader(self, dataset: SegmentationDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset, shuffle=False)
