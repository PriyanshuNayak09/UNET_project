"""Stores all project configurations and hyperparameters."""

import torch
SEED = 42

# ==========================================================
# DEVICE CONFIGURATION
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# DATASET PATHS
# ==========================================================
TRAIN_IMG_DIR = "UNET_data/train_images"
TRAIN_MASK_DIR = "UNET_data/train_masks"

VAL_IMG_DIR = "UNET_data/val_images"
VAL_MASK_DIR = "UNET_data/val_masks"

# ==========================================================
# TRAINING HYPERPARAMETERS
# ==========================================================
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 10

# Number of CPU processes used for loading data
NUM_WORKERS = 0

# Speeds up CPU → GPU memory transfer
PIN_MEMORY = True

# ==========================================================
# IMAGE SIZE
# ==========================================================
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240

# ==========================================================
# MODEL CONFIGURATION
# ==========================================================
IN_CHANNELS = 3
OUT_CHANNELS = 1
FEATURES = [64, 128, 256, 512]

# ==========================================================
# CHECKPOINTS SETTINGS
# ==========================================================
CHECKPOINT_DIR = "checkpoints"

CHECKPOINT_NAME = "best_model.pth"

CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/{CHECKPOINT_NAME}"

SAVE_BEST_ONLY = True

# ==========================================================
# RESULTS
# ==========================================================
RESULTS_DIR = "results"
COMPARISON_DIR = f"{RESULTS_DIR}/comparison"
LOSS_CURVE = f"{RESULTS_DIR}/loss_curve.png"
METRICS_CURVE = f"{RESULTS_DIR}/metrics_curve.png"
SAVE_PREDICTIONS = True
SAVE_PLOTS = True



# ==========================================================
# NORMALIZATION (ImageNet Statistics)
# ==========================================================
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
