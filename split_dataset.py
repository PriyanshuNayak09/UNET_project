"""
Randomly splits the Carvana dataset into training and validation sets.

Training Images  : 4588
Validation Images: 500
"""

import os
import random
import shutil

# ==========================================================
# Configuration
# ==========================================================
SEED = 42

TRAIN_IMAGES_DIR = "UNET_data/train_images"
TRAIN_MASKS_DIR = "UNET_data/train_masks"
VAL_IMAGES_DIR = "UNET_data/val_images"
VAL_MASKS_DIR = "UNET_data/val_masks"
NUM_VALIDATION_IMAGES = 500

# ==========================================================
# Create Validation Folders
# ==========================================================
os.makedirs(VAL_IMAGES_DIR, exist_ok=True)
os.makedirs(VAL_MASKS_DIR, exist_ok=True)

# ==========================================================
# Set Random Seed
# ==========================================================
random.seed(SEED)

# ==========================================================
# Read All Training Images
# ==========================================================
all_images = [
    file
    for file in os.listdir(TRAIN_IMAGES_DIR)
    if file.endswith(".jpg")
]

print(f"Total Images Found : {len(all_images)}")

# ==========================================================
# Randomly Select Validation Images
# ==========================================================
validation_images = random.sample(
    all_images,
    NUM_VALIDATION_IMAGES,
)

# ==========================================================
# Move Images and Corresponding Masks
# ==========================================================
for image_name in validation_images:
    image_src = os.path.join(
        TRAIN_IMAGES_DIR,
        image_name,
    )

    image_dst = os.path.join(
        VAL_IMAGES_DIR,
        image_name,
    )

    mask_name = image_name.replace(
        ".jpg",
        "_mask.gif",
    )

    mask_src = os.path.join(
        TRAIN_MASKS_DIR,
        mask_name,
    )

    mask_dst = os.path.join(
        VAL_MASKS_DIR,
        mask_name,
    )

    shutil.move(image_src, image_dst)
    shutil.move(mask_src, mask_dst)

# ==========================================================
# Summary
# ==========================================================
print("=" * 50)
print("Dataset Split Completed Successfully!")
print(f"Validation Images Moved : {NUM_VALIDATION_IMAGES}")
print(f"Training Images Left    : {len(all_images) - NUM_VALIDATION_IMAGES}")
print("=" * 50)