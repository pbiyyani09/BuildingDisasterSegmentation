import torch
import pytorch_lightning as pl
from pathlib import Path
import numpy as np
from PIL import Image
from collections import OrderedDict

# Define the color map for visualization
COLOR_MAP = OrderedDict([
    ('unlabeled', (0, 0, 0)),
    ('water', (61, 230, 250)),
    ('building-no-damage', (180, 120, 120)),
    ('building-medium-damage', (235, 255, 7)),
    ('building-major-damage', (255, 184, 6)),
    ('building-total-destruction', (255, 0, 0)),
    ('vehicle', (255, 0, 245)),
    ('road-clear', (140, 140, 140)),
    ('road-blocked', (160, 150, 20)),
    ('tree', (4, 250, 7)),
    ('pool', (255, 235, 0))
])

def decode_segmap(label_mask, num_classes):
    """Converts a segmentation mask (H, W) to a color image (H, W, 3)."""
    color_map_values = np.array(list(COLOR_MAP.values()), dtype=np.uint8)
    r = np.zeros_like(label_mask).astype(np.uint8)
    g = np.zeros_like(label_mask).astype(np.uint8)
    b = np.zeros_like(label_mask).astype(np.uint8)
    
    for class_idx in range(num_classes):
        idx = label_mask == class_idx
        r[idx] = color_map_values[class_idx, 0]
        g[idx] = color_map_values[class_idx, 1]
        b[idx] = color_map_values[class_idx, 2]
        
    rgb_image = np.stack([r, g, b], axis=2)
    return rgb_image

class ValidationImageLogger(pl.Callback):
    def __init__(self, save_dir, num_samples=4):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.num_samples = num_samples
        self.color_map = list(COLOR_MAP.values())
        # Denormalization transform for viewing images
        self.inv_normalize = np.array([
            [1/0.229, 1/0.224, 1/0.225]
        ]).T
        self.inv_mean = np.array([
            [-0.485/0.229, -0.456/0.224, -0.406/0.225]
        ]).T

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Log only on the first batch of each validation epoch
        if batch_idx > 10:
            return

        images, labels = batch
        preds = torch.argmax(outputs['preds'], dim=1) 

        for i in range(min(self.num_samples, len(images))):
            # --- Prepare paths ---
            # Assume image paths are available in the dataloader if needed, or use index
            img_id = f"epoch_{trainer.current_epoch}_batch_{batch_idx}_img_{i}"
            result_dir = self.save_dir / img_id
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # --- De-normalize and convert original image ---
            img_np = images[i].cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * self.inv_normalize.T) + self.inv_mean.T
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

            # --- Decode prediction and label masks ---
            pred_mask = decode_segmap(preds[i].cpu().numpy(), pl_module.hparams.model_config['num_classes'])
            label_mask = decode_segmap(labels[i].cpu().numpy(), pl_module.hparams.model_config['num_classes'])

            # --- Save images ---
            Image.fromarray(img_np).save(result_dir / "image.png")
            Image.fromarray(label_mask).save(result_dir / "ground_truth.png")
            Image.fromarray(pred_mask).save(result_dir / "prediction.png")