import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import segmentation_models_pytorch as smp
import pytorch_lightning as pl


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        arch: str,
        encoder_name: str,
        encoder_weights: str,
        in_channels: int,
        out_classes: int,
        criterion: torch.nn.Module,
        lr: float,
        batch_size: int,
        epochs: int,
        steps_per_epoch: int,
        mode: str = "binary",
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        self.loss_fn = criterion
        self.mode = mode

    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)

    def shared_step(self, batch: dict, stage: str) -> Tensor:
        image, mask = batch["image"], batch["mask"].float()
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        if self.mode == "binary":
            pred_mask = (logits_mask.sigmoid() > 0.5).long()
        else:
            pred_mask = logits_mask.softmax(dim=1).argmax(dim=1, keepdim=True)

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask, mask.long(), mode=self.mode)
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        log_kwargs: dict = dict(prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.hparams.batch_size)
        self.log(f"{stage}_loss", loss, **log_kwargs)
        self.log(f"{stage}_per_image_iou", per_image_iou, **log_kwargs)
        self.log(f"{stage}_dataset_iou", dataset_iou, **log_kwargs)

        return loss

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self.shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self.shared_step(batch, "valid")

    def test_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self.shared_step(batch, "test")

    def configure_optimizers(self) -> dict:
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                steps_per_epoch=self.hparams.steps_per_epoch,
                epochs=self.hparams.epochs,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
