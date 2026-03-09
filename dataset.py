from pathlib import Path
from typing import Optional

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])


class SegmentationDataset(Dataset):
    SUPPORTED_EXTS = ("*.png", "*.jpg", "*.jpeg")

    def __init__(self, image_dir: str, mask_dir: str, transform: Optional[A.Compose] = None) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        self.images = sorted([
            p.name
            for ext in self.SUPPORTED_EXTS
            for p in self.image_dir.glob(ext)
        ])

        if not self.images:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_dir / self.images[idx]
        mask_path = self.mask_dir / self.images[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(f"Failed to load: {img_path} or {mask_path}")

        mask = mask / 255.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return {"image": image, "mask": mask.unsqueeze(0)}
