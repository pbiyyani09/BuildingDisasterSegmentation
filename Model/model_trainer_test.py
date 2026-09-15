import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Metric
from .segmentor_model_test import EnhancedUNetNew

# --- NEW: Import the compound loss function ---
import numpy as np
from PIL import Image
from pathlib import Path

# --- NEW: Loss Functions ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        # Convert logits to probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # Create one-hot encoding for targets
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Mask out ignored pixels
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float().unsqueeze(1)
            inputs = inputs * mask
            targets_one_hot = targets_one_hot * mask
        
        # Compute Dice coefficient
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (
            inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3)) + self.smooth
        )
        
        return 1 - dice_score.mean()

class CompoundLoss(nn.Module):
    def __init__(self, class_weights=None, alpha=0.25, gamma=2.0, ignore_index=255):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        # Weighted combination
        total_loss = 0.5 * ce + 0.3 * focal + 0.2 * dice
        return total_loss, {'ce': ce, 'focal': focal, 'dice': dice}

# --- Helper Function for Metrics (unchanged) ---
def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection.float(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def poly_lr_lambda(current_step: int, max_steps: int, power: float):
    """Lambda function for the poly learning rate policy."""
    return (1 - current_step / max_steps) ** power

# --- Custom Metric for Segmentation (unchanged) ---
class SegmentationMetric(Metric):
    """
    Calculates Mean IoU, Overall Accuracy, and Per-Class Accuracy for segmentation.
    """
    def __init__(self, num_classes, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.add_state("total_intersection", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("total_union", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("total_target", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds are logits (B, C, H, W), target is (B, H, W)
        preds = torch.argmax(preds, dim=1)
        
        intersection, union, target_area = intersectionAndUnionGPU(preds, target, self.num_classes, self.ignore_index)
        self.total_intersection += intersection.to(self.device)
        self.total_union += union.to(self.device)
        self.total_target += target_area.to(self.device)

    def compute(self):
        # Mean IoU
        iou = self.total_intersection / (self.total_union + 1e-6)
        mean_iou = torch.mean(iou)
        
        # Overall Pixel Accuracy
        accuracy_all = torch.sum(self.total_intersection) / (torch.sum(self.total_target) + 1e-6)
        
        # Per-Class Accuracy
        accuracy_class = self.total_intersection / (self.total_target + 1e-6)
        
        return {
            "mIoU": mean_iou,
            "PixelAcc": accuracy_all,
            "ClassAcc": accuracy_class
        }

# --- UPDATED: PyTorch Lightning Module ---
class RescueNetLightningNew(pl.LightningModule):
    def __init__(self, model_config: dict, training_config: dict, class_weights=None):
        super().__init__()
        self.save_hyperparameters() # Saves configs to the checkpoint
        
        self.model = EnhancedUNetNew(
            img_ch=model_config['in_channels'], 
            output_ch=model_config['num_classes']
        )
        
        # NEW: Use compound loss with class weights
        self.criterion = CompoundLoss(
            class_weights=class_weights,
            alpha=training_config.get('focal_alpha', 0.25),
            gamma=training_config.get('focal_gamma', 2.0),
            ignore_index=model_config.get('ignore_index', 255)
        )
        
        # Metrics (unchanged)
        self.train_metrics = SegmentationMetric(model_config['num_classes'])
        self.val_metrics = SegmentationMetric(model_config['num_classes'])
        self.test_metrics = SegmentationMetric(model_config['num_classes'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        # Model returns two outputs in training mode
        if self.model.training and self.model.use_aux_loss:
            main_output, aux_output = self.forward(images)
            
            # NEW: Use compound loss
            main_loss, main_loss_dict = self.criterion(main_output, labels)
            aux_loss, aux_loss_dict = self.criterion(aux_output, labels)
            
            total_loss = main_loss + self.hparams.training_config['aux_loss_weight'] * aux_loss
            
            # NEW: Log detailed losses
            self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train/main_loss', main_loss, on_step=True, on_epoch=True)
            self.log('train/aux_loss', aux_loss, on_step=True, on_epoch=True)
            
            # NEW: Log individual loss components
            for loss_name, loss_value in main_loss_dict.items():
                self.log(f'train/main_{loss_name}', loss_value, on_step=True, on_epoch=True)
        else:
            main_output = self.forward(images)
            total_loss, loss_dict = self.criterion(main_output, labels)
            
            self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
            # NEW: Log individual loss components
            for loss_name, loss_value in loss_dict.items():
                self.log(f'train/{loss_name}', loss_value, on_step=True, on_epoch=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
    
        # In eval mode, model returns only the main output
        main_output = self.forward(images)
        
        # NEW: Use compound loss
        loss, loss_dict = self.criterion(main_output, labels)
    
        self.val_metrics.update(main_output, labels)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # NEW: Log individual loss components
        for loss_name, loss_value in loss_dict.items():
            self.log(f'val/{loss_name}', loss_value, on_step=False, on_epoch=True)
    
        return {'preds': main_output}

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log('val/mIoU', metrics['mIoU'], on_epoch=True)
        self.log('val/PixelAcc', metrics['PixelAcc'], on_epoch=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        images, labels = batch
        main_output = self.forward(images)
        
        # NEW: Use compound loss
        loss, loss_dict = self.criterion(main_output, labels)
        
        self.test_metrics.update(main_output, labels)
        self.log('test/loss', loss, on_epoch=True)
        
        # NEW: Log individual loss components
        for loss_name, loss_value in loss_dict.items():
            self.log(f'test/{loss_name}', loss_value, on_epoch=True)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log('test/mIoU', metrics['mIoU'], on_epoch=True)
        self.log('test/PixelAcc', metrics['PixelAcc'], on_epoch=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        config = self.hparams.training_config
        
        # NEW: Use AdamW instead of SGD for better convergence
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # NEW: Cosine annealing with warm restarts instead of poly LR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get('restart_epochs', 10),
            T_mult=1,
            eta_min=config.get('min_lr', 1e-7)
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }