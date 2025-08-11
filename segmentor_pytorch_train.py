import os
import time
import argparse
import datetime
import numpy as np
import toml
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Model import TEEDInspiredAttUNet, AttU_Net  # Your model
from Data_Processing import RescueNetDataModule  # Your dataloader

# Accelerator import
from accelerate import Accelerator
from accelerate.utils import set_seed

# Utility functions for IoU and poly learning rate
def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    """Calculate intersection and union on GPU"""
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()

def enet_weighing(dataloader, num_classes, c=1.02):
    """Computes class weights using the ENet method"""
    class_count = 0
    total = 0
    for image, label in dataloader:
        label = label.cpu().numpy()
        flat_label = label.flatten()
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size
    
    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))
    
    return class_weights

def check_makedirs(dir_name):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def save_checkpoint(model, optimizer, scheduler, epoch, best_iou, checkpoint_path, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'best_iou': best_iou
    }
    
    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_path, 'model_best.pth'))
    torch.save(checkpoint, os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch}.pth'))

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        return 0, 0  # Start from epoch 0
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return checkpoint['epoch'], checkpoint['best_iou']

def get_current_lr(optimizer):
    """Get current learning rate"""
    return optimizer.param_groups[0]['lr']

def validate(model, val_loader, num_classes, accelerator):
    """Validation function"""
    model.eval()
    intersection_sum = 0
    union_sum = 0
    target_sum = 0
    
    with torch.no_grad():
        for image, target in val_loader:
            output = model(image)
            
            # Handle the case where model returns tuple (main_output, aux_output)
            if isinstance(output, tuple):
                output = output[0]
                
            prediction = output.argmax(dim=1)
            intersection, union, target_area = intersectionAndUnionGPU(
                prediction, target, num_classes)
                
            intersection_sum += intersection
            union_sum += union
            target_sum += target_area
    
    # Calculate IoU for each class and mean IoU
    iou_class = intersection_sum / (union_sum + 1e-10)
    mean_iou = iou_class.mean().item()
    
    return mean_iou, iou_class

def train(config, args):
    """Main training function"""
    # Initialize accelerator
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)
    accelerator.print(f"Device: {accelerator.device}")
    
    # Set seed for reproducibility
    set_seed(config['training']['seed'])
    
    # Create output directories
    experiment_name = f"{config['experiment']['name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(config['experiment']['output_dir']) / experiment_name
    checkpoint_dir = output_dir / "checkpoints"
    tensorboard_dir = output_dir / "tensorboard"
    check_makedirs(output_dir)
    check_makedirs(checkpoint_dir)
    check_makedirs(tensorboard_dir)
    
    # Save config for reproducibility
    with open(output_dir / "config.toml", "w") as f:
        toml.dump(config, f)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    # Initialize data module
    data_module = RescueNetDataModule(
        data_dir=config['dataset']['data_dir'],
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        image_size=(config['dataset']['image_size'], config['dataset']['image_size'])
    )
    data_module.setup()
    
    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Calculate class weights using ENet method
    num_classes = config['model']['num_classes']
    class_weights = enet_weighing(train_loader, num_classes)
    class_weights = torch.from_numpy(class_weights).float()
    accelerator.print(f"Class weights: {class_weights}")
    
    # Initialize model
    if config['model']['model_type'] == "TEEDInspiredAttUNet":
        model = TEEDInspiredAttUNet(
            img_ch=config['model']['input_channels'],
            output_ch=config['model']['num_classes']
        )
    elif config['model']['model_type'] == "AttU_Net":
        model = AttU_Net(
            img_ch=config['model']['input_channels'],
            output_ch=config['model']['num_classes']
        )
    else:
        raise ValueError(f"Invalid model type: {config['model']['model_type']}")
    
    # IMPORTANT: Move class weights to the same device as the model
    class_weights = class_weights.to(accelerator.device)
    
    # Define loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Define optimizer - SGD with momentum as specified
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['optimizer']['base_lr'],
        momentum=config['optimizer'].get('momentum', 0.9),
        weight_decay=config['optimizer'].get('weight_decay', 1e-4)
    )
    
    # Prepare for accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Load checkpoint if it exists
    start_epoch, best_iou = 0, 0
    if args.resume and os.path.exists(args.resume):
        accelerator.print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_iou = load_checkpoint(model, optimizer, None, args.resume)
        accelerator.print(f"Resumed from epoch {start_epoch} with best IoU {best_iou}")
    
    # Total number of iterations for poly learning rate
    total_iterations = len(train_loader) * config['training']['epochs']
    current_iteration = start_epoch * len(train_loader)
    
    # Training loop
    accelerator.print("Starting training...")
    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        epoch_loss = 0
        epoch_intersection_sum = torch.zeros(num_classes).cuda()
        epoch_union_sum = torch.zeros(num_classes).cuda()
        epoch_target_sum = torch.zeros(num_classes).cuda()
        
        start_time = time.time()
        
        for step, (images, targets) in enumerate(train_loader):
            # Update learning rate based on poly policy
            current_iteration = epoch * len(train_loader) + step
            current_lr = poly_learning_rate(
                config['optimizer']['base_lr'], 
                current_iteration, 
                total_iterations,
                power=config['optimizer'].get('power', 0.9)
            )
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Forward pass
            outputs = model(images)
            
            # Handle auxiliary loss if model returns tuple
            if isinstance(outputs, tuple):
                main_output, aux_output = outputs
                main_loss = criterion(main_output, targets)
                aux_loss = criterion(aux_output, targets)
                loss = main_loss + 0.4 * aux_loss  # Common aux loss weight
            else:
                main_output = outputs
                loss = criterion(main_output, targets)
            
            # Backward and optimize
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Calculate IoU for this step
            with torch.no_grad():
                prediction = main_output.argmax(dim=1)
                intersection, union, target_area = intersectionAndUnionGPU(
                    prediction, targets, num_classes)
                
                epoch_intersection_sum += intersection
                epoch_union_sum += union
                epoch_target_sum += target_area
                
                # Calculate step IoU
                step_iou_class = intersection / (union + 1e-10)
                step_miou = step_iou_class.mean().item()
            
            # Log step metrics
            if step % config['logging']['log_interval'] == 0:
                accelerator.print(
                    f"Epoch [{epoch+1}/{config['training']['epochs']}], "
                    f"Step [{step+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"IoU: {step_miou:.4f}, "
                    f"LR: {current_lr:.6f}"
                )
                
                # Log to tensorboard (step metrics)
                writer.add_scalar('Train/StepLoss', loss.item(), current_iteration)
                writer.add_scalar('Train/StepMIoU', step_miou, current_iteration)
                writer.add_scalar('Train/LearningRate', current_lr, current_iteration)
                
                for i in range(num_classes):
                    writer.add_scalar(f'Train/StepIoU_Class{i}', step_iou_class[i].item(), current_iteration)
        
        # Calculate epoch metrics
        epoch_loss = epoch_loss / len(train_loader)
        epoch_iou_class = epoch_intersection_sum / (epoch_union_sum + 1e-10)
        epoch_miou = epoch_iou_class.mean().item()
        
        # Validation
        val_miou, val_iou_class = validate(model, val_loader, num_classes, accelerator)
        
        # Check if this is the best model
        is_best = val_miou > best_iou
        if is_best:
            best_iou = val_miou
        
        # Save checkpoint
        if accelerator.is_main_process:
            save_checkpoint(
                model, optimizer, None, epoch + 1, best_iou, 
                checkpoint_dir, is_best=is_best
            )
        
        # Log epoch metrics
        epoch_time = time.time() - start_time
        accelerator.print(
            f"Epoch [{epoch+1}/{config['training']['epochs']}] completed in {epoch_time:.2f}s, "
            f"Train Loss: {epoch_loss:.4f}, Train mIoU: {epoch_miou:.4f}, "
            f"Val mIoU: {val_miou:.4f}, Best mIoU: {best_iou:.4f}"
        )
        
        # Log to tensorboard (epoch metrics)
        writer.add_scalar('Train/EpochLoss', epoch_loss, epoch + 1)
        writer.add_scalar('Train/EpochMIoU', epoch_miou, epoch + 1)
        writer.add_scalar('Val/MIoU', val_miou, epoch + 1)
        
        for i in range(num_classes):
            writer.add_scalar(f'Train/EpochIoU_Class{i}', epoch_iou_class[i].item(), epoch + 1)
            writer.add_scalar(f'Val/IoU_Class{i}', val_iou_class[i].item(), epoch + 1)
    
    writer.close()
    accelerator.print(f"Training finished. Best mIoU: {best_iou:.4f}")
    accelerator.end_training()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RescueNet Semantic Segmentation Training")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
                       help="Whether to use mixed precision training")
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU")
    args = parser.parse_args()
    
    config_path = Path("./Options/config.toml").absolute()
    
    # Load config file
    config = toml.load(str(config_path))
    
    # Start training
    train(config, args)

if __name__ == "__main__":
    main()