from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import numpy as np
from PIL import Image

# Recommended: Use Albumentations for synchronous image/mask transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RescueNetDataset(Dataset):
    """
    Internal Dataset for RescueNet, designed to work with Albumentations.

    Args:
        image_paths (list): List of paths to images.
        label_paths (list): List of paths to label masks.
        transform (callable, optional): Albumentations transform pipeline.
    """
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load image and label
        img_path = self.image_paths[index]
        lbl_path = self.label_paths[index]

        # Albumentations works best with numpy arrays
        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(lbl_path))

        # Apply the unified transform pipeline
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
        
        # Ensure the label is of type Long
        label = label.long()

        return image, label

class RescueNetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the RescueNet Dataset using Albumentations.

    Args:
        data_dir (str): Root directory of the dataset.
        batch_size (int, optional): The batch size for dataloaders. Defaults to 32.
        num_workers (int, optional): Number of workers for dataloaders. Defaults to 4.
        image_size (tuple, optional): The size to resize images to (height, width). Defaults to (256, 256).
    """
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, image_size: tuple = (256, 256)):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        # Define transformations using Albumentations
        # 1. Training transforms: Spatial + Pixel-level augmentations
        self.train_transform = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            # Spatial augmentations applied to both image and mask
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.7),
            # Pixel-level augmentations applied ONLY to the image
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Convert to PyTorch Tensor
            ToTensorV2(),
        ])

        # 2. Validation/Test transforms: Only resizing, normalization, and tensor conversion
        self.val_test_transform = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def setup(self, stage: str = None):
        """Finds file paths and creates Datasets."""
        if stage == 'fit' or stage is None:
            train_image_paths, train_label_paths = self._get_file_paths("train")
            val_image_paths, val_label_paths = self._get_file_paths("val")

            self.train_dataset = RescueNetDataset(
                train_image_paths, train_label_paths, transform=self.train_transform
            )
            self.val_dataset = RescueNetDataset(
                val_image_paths, val_label_paths, transform=self.val_test_transform
            )
        
        if stage == 'test' or stage is None:
            test_image_paths, test_label_paths = self._get_file_paths("test")
            self.test_dataset = RescueNetDataset(
                test_image_paths, test_label_paths, transform=self.val_test_transform
            )

    def _get_file_paths(self, mode: str):
        """Helper to find and pair image and label files."""
        img_dir = self.data_dir / mode / f"{mode}-org-img"
        lbl_dir = self.data_dir / mode / f"{mode}-label-img"
        
        image_paths = sorted([p for p in img_dir.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        label_paths = []
        valid_image_paths = []
        for img_path in image_paths:
            expected_label_path = lbl_dir / f"{img_path.stem}_lab.png"
            if expected_label_path.exists():
                label_paths.append(expected_label_path)
                valid_image_paths.append(img_path)
        
        print(f"Found {len(valid_image_paths)} paired images and labels for mode '{mode}'.")
        return valid_image_paths, label_paths

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
