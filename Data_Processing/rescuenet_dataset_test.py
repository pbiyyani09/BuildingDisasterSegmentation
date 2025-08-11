from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import numpy as np
from torch import zeros, clamp 
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

class RescueNetDataModuleNew(pl.LightningDataModule):
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
        self.class_weights = None
        # Define transformations using Albumentations
        # 1. Training transforms: Spatial + Pixel-level augmentations
        # Updated rescuenet_dataset.py
        self.train_transform = A.Compose([
            A.Resize(height=256, width=256),  # Keep current resolution
            
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.7),
            
            # Lighting/weather effects (crucial for disaster imagery)
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.7),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.6),
            
            # Weather conditions
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15),
            A.RandomRain(slant_lower=-5, slant_upper=5, drop_length=10, p=0.15),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.25),
            
            # Noise (simulates camera quality variations)
            A.OneOf([
                A.GaussNoise(var_limit=(5, 25), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=0.3),
            ], p=0.4),
            
            # Subtle blur effects
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.2),
            ], p=0.25),
    
            # Dropout for robustness
            A.CoarseDropout(max_holes=4, max_height=20, max_width=20, fill_value=0, p=0.3),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        # 2. Validation/Test transforms: Only resizing, normalization, and tensor conversion
        self.val_test_transform = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


    def compute_class_weights(self, num_classes=11):
        """Compute class weights for handling imbalanced dataset."""
        class_counts = zeros(num_classes)
        
        # Count pixels for each class
        label_dir = self.data_dir / "train" / "train-label-img"
        for label_path in label_dir.glob("*.png"):
            label = np.array(Image.open(label_path))
            unique, counts = np.unique(label, return_counts=True)
            
            for class_id, count in zip(unique, counts):
                if class_id < num_classes:
                    class_counts[class_id] += count
        
        # Calculate inverse frequency weights
        total_pixels = class_counts.sum()
        class_weights = total_pixels / (num_classes * class_counts)
        
        # Smooth extreme weights
        class_weights = clamp(class_weights, min=0.1, max=10.0)
        
        print("Class distribution:")
        for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
            print(f"Class {i}: {count.item():>10.0f} pixels, weight: {weight.item():.3f}")
        
        return class_weights
    

    def setup(self, stage: str = None):
        """Finds file paths and creates Datasets."""
        if stage == 'fit' or stage is None:
            # Compute class weights during setup
            self.class_weights = self.compute_class_weights()
            
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True,persistent_workers=True,
        prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True,persistent_workers=True,
        prefetch_factor=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True,persistent_workers=True,
        prefetch_factor=2)
