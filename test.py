import toml
import torch
from torchvision import transforms
import pytorch_lightning as pl
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from collections import OrderedDict

# Import your custom modules
# Make sure your __init__.py files are configured correctly
from Model import RescueNetLightning
from Model.segmentor_model import * # Needed for Lightning to load the model class

# --- Configuration ---
# Load settings from your TOML file
CONFIG = toml.load("./Options/config.toml")
MODEL_CONFIG = CONFIG['MODEL']
DATA_CONFIG = CONFIG['DATA']

# Paths
CHECKPOINT_PATH = "./Training_Logs/RescueNet_Training/version_0/checkpoints/best-mIoU-162-0.5995.ckpt" # <--- IMPORTANT: SET YOUR CHECKPOINT PATH
IMAGE_DIR = "./Data/RescueNet/val/val-org-img"  # <--- A folder with images you want to test
OUTPUT_DIR = "./Val_Results"

# Tiled Inference Parameters
TILE_SIZE = 256  # The size of the image patches
OVERLAP = 64     # The overlap between patches

# --- Helper Functions ---
# Color map for visualizing the segmentation output
COLOR_MAP = OrderedDict([
    ('unlabeled', (0, 0, 0)), ('water', (61, 230, 250)),
    ('building-no-damage', (180, 120, 120)), ('building-medium-damage', (235, 255, 7)),
    ('building-major-damage', (255, 184, 6)), ('building-total-destruction', (255, 0, 0)),
    ('vehicle', (255, 0, 245)), ('road-clear', (140, 140, 140)),
    ('road-blocked', (160, 150, 20)), ('tree', (4, 250, 7)),
    ('pool', (255, 235, 0))
])

def decode_segmap(label_mask, num_classes):
    """Converts a segmentation mask (H, W) to a color image (H, W, 3)."""
    color_map_values = np.array(list(COLOR_MAP.values()), dtype=np.uint8)
    rgb_image = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8)
    for class_idx in range(num_classes):
        idx = label_mask == class_idx
        rgb_image[idx] = color_map_values[class_idx]
    return rgb_image

def process_image_with_tiling(model, img_tensor, tile_size=512, overlap=64, num_classes=11):
    """
    Processes a large image tensor by breaking it into overlapping tiles,
    running inference on each, and blending the results.

    Args:
        model (nn.Module): The segmentation model.
        img_tensor (torch.Tensor): The full image tensor (1, C, H, W), normalized.
        tile_size (int): The size of each square tile.
        overlap (int): The pixel overlap between adjacent tiles.
        num_classes (int): The number of output classes from the model.

    Returns:
        torch.Tensor: The final predicted class map for the full image (H, W).
    """
    device = img_tensor.device
    b, c, h, w = img_tensor.shape
    
    # If the image is smaller than the tile size, process it directly
    if h <= tile_size and w <= tile_size:
        with torch.no_grad():
            logits = model(img_tensor)
        return torch.argmax(logits, dim=1).squeeze(0)

    stride = tile_size - overlap
    
    # Prepare tensors for storing the blended results
    full_logits = torch.zeros((1, num_classes, h, w), device=device)
    weight_map = torch.zeros((1, 1, h, w), device=device)

    # Create a Gaussian weighting map for smooth blending
    x = torch.linspace(-1, 1, tile_size)
    y = torch.linspace(-1, 1, tile_size)
    xv, yv = torch.meshgrid(x, y, indexing='ij')
    gaussian_weight = torch.exp(-((xv**2 + yv**2) / 0.5)).unsqueeze(0).unsqueeze(0).to(device)

    # Iterate over the image with overlapping tiles
    for y in range(0, h - tile_size + stride, stride):
        for x in range(0, w - tile_size + stride, stride):
            # Ensure we don't go past the image boundaries
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = y_end - tile_size
            x_start = x_end - tile_size
            
            tile = img_tensor[:, :, y_start:y_end, x_start:x_end]
            
            with torch.no_grad():
                tile_logits = model(tile)

            # Add the weighted logits to the full output tensor
            full_logits[:, :, y_start:y_end, x_start:x_end] += tile_logits * gaussian_weight
            weight_map[:, :, y_start:y_end, x_start:x_end] += gaussian_weight
            
    # Normalize the logits by the accumulated weights
    final_logits = full_logits / (weight_map + 1e-8)
    
    return torch.argmax(final_logits, dim=1).squeeze(0)


def test():
    """Main testing function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model from Checkpoint
    try:
        model = RescueNetLightning.load_from_checkpoint(
            checkpoint_path=CHECKPOINT_PATH,
            map_location=device
        )
        model.eval()
        model.to(device)
        print("Model loaded successfully from checkpoint.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at '{CHECKPOINT_PATH}'. Please update the path.")
        return

    # 2. Prepare Image Transformations (must match validation transforms)
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Find and Process Images
    image_files = list(Path(IMAGE_DIR).glob('*.png')) + list(Path(IMAGE_DIR).glob('*.jpg'))
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    print(f"Found {len(image_files)} images to test in '{IMAGE_DIR}'.")

    for img_file in tqdm(image_files, desc="Processing Images"):
        # Load and transform image
        original_image = Image.open(img_file).convert("RGB")
        img_tensor = image_transform(original_image).unsqueeze(0).to(device)

        # Run inference using the tiling method
        prediction_map = process_image_with_tiling(
            model, img_tensor, TILE_SIZE, OVERLAP, MODEL_CONFIG['num_classes']
        )
        
        # Convert prediction to a color mask
        pred_np = prediction_map.cpu().numpy()
        pred_mask_rgb = decode_segmap(pred_np, MODEL_CONFIG['num_classes'])

        # Create overlay image
        original_np = np.array(original_image)
        overlay = cv2.addWeighted(original_np, 0.6, pred_mask_rgb, 0.4, 0)

        # Save results
        save_folder = output_path / img_file.stem
        save_folder.mkdir(exist_ok=True)
        
        Image.fromarray(original_np).save(save_folder / "image.png")
        Image.fromarray(pred_mask_rgb).save(save_folder / "segmentation_mask.png")
        Image.fromarray(overlay).save(save_folder / "overlay.png")

    print(f"--- Testing complete. Results saved in '{OUTPUT_DIR}'. ---")


if __name__ == '__main__':
    test()