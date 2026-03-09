# Segmentation Baseline

Universal PyTorch Lightning template for binary / multiclass image segmentation. Swap encoder, architecture, and task mode via a single config file.

## Stack

- [PyTorch Lightning](https://lightning.ai/) — training loop, callbacks, logging
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) — encoder-decoder architectures
- [Albumentations](https://albumentations.ai/) — augmentations
- [TensorBoard](https://www.tensorflow.org/tensorboard) — experiment tracking
- [Poetry](https://python-poetry.org/) — dependency management

## Project structure

```
baseline/
├── cfg.yaml            # all hyperparameters and paths
├── train.py            # training entry point
├── infer.py            # inference / test evaluation
├── model.py            # LightningModule (SegmentationModel)
├── datamodule.py       # LightningDataModule
├── dataset.py          # Dataset + Albumentations transforms
├── callbacks.py        # ModelCheckpoint, EarlyStopping, LR monitor
├── loss.py             # BCEDiceLoss
├── config_loader.py    # YAML config loader
└── utils.py            # visualization helpers
```

## Dataset structure

```
data/splits/
├── train/
│   ├── image/   ← grayscale .png / .jpg images
│   └── mask/    ← binary masks (same filenames, values 0 / 255)
├── val/
│   ├── image/
│   └── mask/
└── test/
    ├── image/
    └── mask/
```

## Quick start

```bash
# install dependencies
poetry install

# set DATASET_PATH in cfg.yaml, then train
python train.py

# evaluate on test set
python infer.py

# open TensorBoard
tensorboard --logdir experiments/tb_logs
```

## Configuration (`cfg.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `DATASET_PATH` | `./data/splits` | Root path to dataset splits |
| `ARCH` | `unetplusplus` | SMP architecture |
| `ENCODER_NAME` | `resnext50_32x4d` | Encoder backbone |
| `ENCODER_WEIGHTS` | `ssl` | Pretrained weights (`imagenet`, `ssl`, `swsl`, `null`) |
| `IN_CHANNELS` | `1` | Input image channels |
| `OUT_CLASSES` | `1` | Number of output classes |
| `MODE` | `binary` | `binary` / `multiclass` / `multilabel` |
| `LR` | `0.0001` | Peak learning rate (OneCycleLR) |
| `BATCH_SIZE` | `4` | Batch size |
| `EPOCHS` | `50` | Max epochs |
| `SEED` | `42` | Global random seed |
| `PRECISION` | `32` | `32` / `16-mixed` / `bf16-mixed` |
| `ACCELERATOR` | `gpu` | `gpu` / `cpu` |
| `NUM_WORKERS` | `4` | DataLoader workers |
| `EXPERIMENT` | `Unet++` | Experiment name (used in checkpoint dir and TensorBoard) |
| `CKPT_PATH` | `null` | Checkpoint to resume from, `null` = train from scratch |

## Supported architectures

Any [SMP architecture](https://smp.readthedocs.io/en/latest/models.html): `unet`, `unetplusplus`, `deeplabv3`, `deeplabv3plus`, `fpn`, `pspnet`, `pan`, `linknet`, `manet`.

Any [timm](https://github.com/huggingface/pytorch-image-models) / SMP encoder: `resnet34`, `resnext50_32x4d`, `efficientnet-b4`, `mit_b2`, `tu-convnext_base`, etc.

## Loss

`BCEDiceLoss = BCEWithLogitsLoss + (1 − DiceScore)` — standard combination for binary segmentation.

## Metrics

| Metric | Description |
|--------|-------------|
| `*_loss` | BCEDice loss |
| `*_per_image_iou` | mean IoU computed per image |
| `*_dataset_iou` | global IoU across the full split |

Prefix `*` is `train` / `valid` / `test`.

## Callbacks

| Callback | Behaviour |
|----------|-----------|
| `ModelCheckpoint` | saves top-3 checkpoints + `last.ckpt`, monitors `valid_loss` |
| `EarlyStopping` | stops if `valid_loss` doesn't improve by `1e-3` for 5 epochs |
| `LearningRateMonitor` | logs LR to TensorBoard every epoch |
