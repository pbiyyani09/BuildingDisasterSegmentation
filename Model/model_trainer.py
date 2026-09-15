import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Metric
# EnhancedUNet was renamed EnhancedUNetNew in segmentor_model_test.py;
# alias keeps this trainer's constructor call working.
from .segmentor_model_test import EnhancedUNetNew as EnhancedUNet

# --- Helper Function for Metrics ---
# This is the GPU-accelerated IoU calculation you provided
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


# --- Custom Metric for Segmentation ---
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

# --- PyTorch Lightning Module ---

class RescueNetLightning(pl.LightningModule):
    def __init__(self, model_config: dict, training_config: dict):
        super().__init__()
        self.save_hyperparameters() # Saves configs to the checkpoint
        
        self.model = EnhancedUNet(
            img_ch=model_config['in_channels'], 
            output_ch=model_config['num_classes']
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=model_config.get('ignore_index', 255))
        
        # Metrics
        self.train_metrics = SegmentationMetric(model_config['num_classes'])
        self.val_metrics = SegmentationMetric(model_config['num_classes'])
        self.test_metrics = SegmentationMetric(model_config['num_classes'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        # Model returns two outputs in training mode
        main_output, aux_output = self.forward(images)
        
        main_loss = self.criterion(main_output, labels)
        aux_loss = self.criterion(aux_output, labels)
        
        total_loss = main_loss + self.hparams.training_config['aux_loss_weight'] * aux_loss
        
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/main_loss', main_loss, on_step=True, on_epoch=True)
        self.log('train/aux_loss', aux_loss, on_step=True, on_epoch=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
    
        # In eval mode, model returns only the main output
        main_output = self.forward(images) # These are your raw prediction logits
        loss = self.criterion(main_output, labels)
    
        self.val_metrics.update(main_output, labels)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    
        # Add this line to pass the predictions to the callback
        return {'preds': main_output}

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log('val/mIoU', metrics['mIoU'], on_epoch=True)
        self.log('val/PixelAcc', metrics['PixelAcc'], on_epoch=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        images, labels = batch
        main_output = self.forward(images)
        loss = self.criterion(main_output, labels)
        
        self.test_metrics.update(main_output, labels)
        self.log('test/loss', loss, on_epoch=True)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log('test/mIoU', metrics['mIoU'], on_epoch=True)
        self.log('test/PixelAcc', metrics['PixelAcc'], on_epoch=True)
        # You could also log the per-class accuracies here if desired
        self.test_metrics.reset()

    def configure_optimizers(self):
        config = self.hparams.training_config
        
        # 1. Define the SGD optimizer with a high initial learning rate
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=config['learning_rate'], # Make sure this is high, e.g., 0.1
            momentum=0.9,
            weight_decay=config['weight_decay']
        )
        
        # 2. Calculate the total number of training steps (max_iter)
        # This requires knowing the number of batches in your training dataloader
        # We can access it after the dataloaders are set up by the Trainer
        
        # Ensure the trainer has the datamodule or train_dataloader attached
        if self.trainer.train_dataloader is None:
            # This can happen during initial setup, handle it gracefully
            # You might need to set a placeholder value or trigger dataloader setup
            # For simplicity, we'll assume the dataloader is available when this is called by the trainer
            self.trainer.fit_loop.setup_data()

        num_steps_per_epoch = len(self.trainer.train_dataloader)
        max_steps = self.trainer.max_epochs * num_steps_per_epoch

        # 3. Create the scheduler using a lambda function
        # The 'power' value is typically 0.9 as used in many segmentation papers
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: poly_lr_lambda(step, max_steps, power=0.9)
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # IMPORTANT: Update after each training step
                "frequency": 1,
            },
        }