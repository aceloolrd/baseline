from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor


def denormalize(img: Tensor) -> Tensor:
    return (img * 0.5 + 0.5).clamp(0, 1)


def visualize_sample_data(dm) -> None:
    for name, dataset in [("Train", dm.train_dataset), ("Validation", dm.val_dataset)]:
        sample = dataset[0]
        img = denormalize(sample["image"])

        _, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img.squeeze(0).numpy(), cmap="gray")
        axes[0].set_title(f"{name} - Image")
        axes[1].imshow(sample["mask"].squeeze(0).numpy(), cmap="gray")
        axes[1].set_title(f"{name} - Mask")
        plt.show()


def visualize_predictions(model, dataloader, num_images: int = 3, save_dir: str = "predictions") -> None:
    model.eval()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(dataloader):
        if i >= num_images:
            break

        image, mask = batch["image"], batch["mask"]
        with torch.no_grad():
            pred_mask = model(image).sigmoid().cpu().numpy()

        _, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image.squeeze().cpu().numpy(), cmap="gray")
        axes[0].set_title("Input Image")
        axes[1].imshow(mask.squeeze().cpu().numpy(), cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[2].imshow(pred_mask.squeeze(), cmap="gray")
        axes[2].set_title("Predicted Mask")

        plt.savefig(save_path / f"prediction_{i}.png")
        plt.close()

    print(f"Predictions saved to {save_dir}")


def get_last_checkpoint(checkpoint_dir: str | None) -> str | None:
    if checkpoint_dir is None:
        return None

    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None

    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    if not ckpt_files:
        return None

    last_ckpt = max(ckpt_files, key=lambda f: f.stat().st_mtime)
    print(f"Found checkpoint: {last_ckpt}")
    return str(last_ckpt)
