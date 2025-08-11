import toml
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Import your custom modules
# Make sure __init__.py files are set up correctly
from Data_Processing import RescueNetDataModuleNew 
from Model import EnhancedUNetNew, RescueNetLightningNew, ValidationImageLogger

# This is required for the trainer to find the EnhancedUNet class
# when loading the model from a checkpoint.
from Model.segmentor_model_test import *


def train():
    print("--- Starting Training ---")
    # Set tensor core precision for RTX 4080 SUPER
    torch.set_float32_matmul_precision('medium')  # Add this line
    
    """Main training function."""
    # 1. Load Configuration
    config = toml.load("./Options/confignew.toml")

    model_config = config['MODEL']
    training_config = config['TRAINING']
    data_config = config['DATA']
    path_config = config['PATHS']

    print(f"Configuration loaded: {config}")
    # 2. Setup DataModule
    data_module = RescueNetDataModuleNew(**data_config)
    print(f"DataModule initialized with data directory: {data_config['data_dir']}")

    # 3. Setup LightningModule
    lightning_model = RescueNetLightningNew( 
        model_config=model_config,
        training_config=training_config
    )
    print("LightningModule initialized with model and training configurations.")

    # 4. Setup Callbacks
    # Logger for TensorBoard
    logger = TensorBoardLogger(path_config['log_dir'], name="RescueNet_Training")

    # Monitor learning rate
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Save the best model based on validation mIoU
    best_model_checkpoint = ModelCheckpoint(
        monitor='val/mIoU',
        mode='max',
        save_top_k=1,
        filename='best-mIoU-{epoch}-{val/mIoU:.4f}',
        auto_insert_metric_name=False
    )

    # Save the model from the last epoch
    last_model_checkpoint = ModelCheckpoint(filename='last-{epoch}')

    # Save validation prediction images
    image_logger = ValidationImageLogger(save_dir=path_config['debug_image_dir'])

    # 5. Setup Trainer
    trainer = pl.Trainer(
        max_epochs=training_config['epochs'],
        accelerator=training_config['accelerator'],
        devices=training_config['devices'],
        precision=training_config['precision'],
        logger=logger,
        callbacks=[
            lr_monitor,
            best_model_checkpoint,
            last_model_checkpoint,
            image_logger
        ],
        accumulate_grad_batches=2,  # Simulates batch_size * 2 = 16
        gradient_clip_val=1.0,      # Prevents gradient explosion
        gradient_clip_algorithm="norm",  # Use norm-based clipping
    )

    # 6. Start Training
    print("--- Starting Training ---")
    trainer.fit(lightning_model, datamodule=data_module)

    # 7. Start Testing
    print("--- Starting Testing with Best Checkpoint ---")
    # trainer.test will automatically use the best checkpoint saved during training
    trainer.test(datamodule=data_module, ckpt_path='best')


if __name__ == '__main__':
    # Set seed for reproducibility
    pl.seed_everything(42, workers=True)
    train()