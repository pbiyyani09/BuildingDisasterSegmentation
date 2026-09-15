# BuildingDisasterSegmentation

Semantic segmentation of **post-disaster UAV (drone) imagery** into 11 classes, built on the
[RescueNet](https://www.nature.com/articles/s41597-023-02799-4) dataset. The goal is to parse
aerial scenes captured after natural disasters into buildings (by damage level), roads, water,
vehicles, vegetation, and other terrain — the kind of map a first responder would want.

This is a **solo research / portfolio project**: a from-scratch PyTorch (Lightning) implementation
of an attention U-Net with several modern segmentation tricks, trained end-to-end on the full
11-class problem.

## The 11 classes

`unlabeled`, `water`, `building-no-damage`, `building-medium-damage`, `building-major-damage`,
`building-total-destruction`, `vehicle`, `road-clear`, `road-blocked`, `tree`, `pool`.

Note that four of the eleven classes are building-damage grades — a hard, fine-grained distinction
that dominates the difficulty of this dataset.

## What's in the model

The main model (`EnhancedUNetNew` in `Model/segmentor_model_test.py`) is a U-Net variant with:

- **Attention gates** on every skip connection (Oktay-style additive attention).
- **CBAM** channel + spatial attention blocks inside the encoder/decoder path.
- **ASPP** (Atrous Spatial Pyramid Pooling) bottleneck for multi-scale context.
- **Separable recurrent residual conv blocks** (R2U-Net-style RRCNN with depthwise-separable convs)
  to keep the parameter count down.
- **Deep supervision** via an auxiliary segmentation head (`aux_loss_weight = 0.4`).

Training (`Model/model_trainer_test.py`, `RescueNetLightningNew`) uses a **compound loss** to fight
the severe class imbalance:

```
loss = 0.5 * weighted_CE + 0.3 * Focal + 0.2 * Dice
```

where the cross-entropy is weighted by **inverse-frequency (ENet-style) class weights** computed
from the training-set pixel counts (`compute_class_weights`, clamped to `[0.1, 10.0]`).

Inference on large aerial frames (`test.py`) uses **tiled inference with Gaussian-weighted
blending** of overlapping tiles, so tile seams don't show up in the output mask.

An earlier, lighter variant (`TEEDInspiredAttUNet` in `Model/segmentor_model.py`, trained via
`train.py` / `RescueNetLightning`) is also kept in the repo for reference.

## Results (honest)

Trained and evaluated on the **full 11-class** RescueNet set:

| Split      | mIoU (11 classes) |
|------------|-------------------|
| Validation | ~0.40 – 0.45      |
| Test       | ~0.42             |

Per-class IoU is tracked during training/validation (via a custom `SegmentationMetric`), and — as
expected — the rare and visually similar classes (the building-damage grades, `pool`, `vehicle`)
are the weakest.

**Framing / caveats.** These numbers are below the best published RescueNet baselines. That gap is
expected and is not hidden here: this is a single-person project trained at a modest input
resolution (256×256 tiles), with a lightweight separable backbone and limited hyperparameter
tuning and compute. If you see ~0.60 mIoU quoted for "RescueNet segmentation" elsewhere, that is
typically a **reduced 5-class** formulation of the problem, which is a much easier task than the
full 11-class one reported here — the two numbers are not comparable.

## Quickstart

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for training. Install the appropriate `torch` /
`torchvision` build for your CUDA version (see https://pytorch.org/get-started/locally/).

### 2. Get the dataset

Download **RescueNet** from its official source and arrange it like this (this layout is what the
data module in `Data_Processing/rescuenet_dataset_test.py` expects):

```
<data_dir>/
├── train/
│   ├── train-org-img/     # RGB images (.jpg)
│   └── train-label-img/   # label masks (.png, pixel values 0–10)
├── val/
│   ├── val-org-img/
│   └── val-label-img/
└── test/
    ├── test-org-img/
    └── test-label-img/
```

Then point the config at it. In `Options/confignew.toml`, set:

```toml
[DATA]
data_dir = "/absolute/path/to/your/RescueNet"
```

(The committed value is a machine-specific path and must be changed.)

### 3. Train

```bash
python train_test.py
```

This runs the main pipeline (`EnhancedUNetNew` + `RescueNetLightningNew`) using
`Options/confignew.toml`. Checkpoints and TensorBoard logs are written under `./Training_Logs`.

```bash
tensorboard --logdir ./Training_Logs
```

### 4. Run inference

Edit the paths at the top of `test.py` (`CHECKPOINT_PATH`, `IMAGE_DIR`, `OUTPUT_DIR`) to point at a
trained checkpoint and a folder of images, then:

```bash
python test.py
```

Color-coded prediction masks are written to `OUTPUT_DIR`.

## Repository layout

```
Model/
  segmentor_model_test.py   # EnhancedUNetNew (main model: CBAM + ASPP + attention gates)
  model_trainer_test.py     # RescueNetLightningNew (compound loss, metrics) — main trainer
  segmentor_model.py        # TEEDInspiredAttUNet (earlier lightweight variant)
  model_trainer.py          # RescueNetLightning (earlier trainer)
  unet_models.py            # baseline U-Net / AttU-Net / R2U-Net building blocks
  callbacks.py              # ValidationImageLogger
Data_Processing/
  rescuenet_dataset_test.py # RescueNetDataModuleNew (+ class-weight computation) — main data module
  rescuenet_dataset.py      # RescueNetDataModule (earlier data module)
Options/
  confignew.toml            # config for the main (train_test.py) pipeline
  config.toml               # config for the earlier pipeline
train_test.py               # main training entry point
train.py                    # earlier training entry point
test.py                     # tiled Gaussian-blended inference
segmentor_pytorch_train.py  # standalone accelerate-based training loop (alternative)
```

## Status / limitations

- Research code, not a packaged library — entry points are run as scripts from the repo root.
- Dataset is not included; you must download RescueNet and set `data_dir`.
- Results reflect limited compute and tuning (see the caveats above).
