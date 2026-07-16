"""train.py"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader

import UNET_config as config
from UNET_seed import set_seed
from UNET_model import UNET
from UNET_dataset import CarvanaDataset

# -------------------------------------------------
# Set random seed for reproducibility
# -------------------------------------------------
set_seed(config.SEED)

# -------------------------------------------------
# Create folders
# -------------------------------------------------
os.makedirs("checkpoints", exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)
os.makedirs(config.COMPARISON_DIR, exist_ok=True)

#========================================
#Transforms
#========================================
train_transform = A.Compose(
    [
        A.Resize(
            height=config.IMAGE_HEIGHT,
            width=config.IMAGE_WIDTH,
        ),

        A.HorizontalFlip(p=0.5),

        A.Normalize(
            mean=config.MEAN,
            std=config.STD,
            max_pixel_value=255.0,
        ),

        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Resize(
            height=config.IMAGE_HEIGHT,
            width=config.IMAGE_WIDTH,
        ),

        A.Normalize(
            mean=config.MEAN,
            std=config.STD,
            max_pixel_value=255.0,
        ),

        ToTensorV2(),
    ]
)

history = {
    "train_loss": [],
    "val_loss": [],
    "dice": [],
    "iou": [],
}

#========================================
#Functions
#========================================
def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0

    loop = tqdm(loader, leave=True)
    
    for batch_idx, (images, masks) in enumerate(loop):
        images = images.to(device)
        masks = masks.unsqueeze(1).float().to(device)

        predictions = model(images)
        loss = loss_fn(predictions, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    average_loss = running_loss / len(loader)
    return average_loss

def validate(loader, model, loss_fn, device):
    model.eval()
    val_loss = 0.0
    dice_score = 0.0
    iou_score = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.unsqueeze(1).float().to(device)
            predictions = model(images)
            loss = loss_fn(predictions, masks)
            val_loss += loss.item()

            predictions = torch.sigmoid(predictions)
            predictions = (predictions > 0.5).float()
            
            intersection = (predictions * masks).sum()
            union = predictions.sum() + masks.sum() - intersection

            dice = (2 * intersection + 1e-8) / (predictions.sum() + masks.sum() + 1e-8)
            iou = (intersection + 1e-8) / (union + 1e-8)

            dice_score += dice.item()
            iou_score += iou.item()

    avg_loss = val_loss / len(loader)
    avg_dice = dice_score / len(loader)
    avg_iou = iou_score / len(loader)

    model.train()

    return avg_loss, avg_dice, avg_iou

def plot_history(history):
    # ----------------------------
    # Loss Graph
    # ----------------------------
    plt.figure(figsize=(8,5))

    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(config.LOSS_CURVE)
    plt.close()

    # ----------------------------
    # Metrics Graph
    # ----------------------------
    plt.figure(figsize=(8,5))

    plt.plot(history["dice"], label="Dice Score")
    plt.plot(history["iou"], label="IoU")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Dice Score and IoU")
    plt.legend()
    plt.grid(True)

    plt.savefig(config.METRICS_CURVE)
    plt.close()

def save_predictions(loader, model, device):
    """
    Saves comparison images:
    Original Image | Ground Truth | Predicted Mask
    """

    model.eval()
    with torch.no_grad():
        for idx, (images, masks) in enumerate(loader):
            # Save only first 5 validation images
            if idx == 5:
                break

            images = images.to(device)
            masks = masks.to(device)

            # -----------------------------
            # Forward Pass
            # -----------------------------
            predictions = model(images)

            predictions = torch.sigmoid(predictions)
            predictions = (predictions > 0.5).float()

            # -----------------------------
            # Convert tensors to NumPy
            # -----------------------------
            image = images[0].cpu().permute(1, 2, 0).numpy()
            mask = masks[0].cpu().squeeze().numpy()
            prediction = predictions[0].cpu().squeeze().numpy()

            # -----------------------------
            # Undo normalization
            # -----------------------------
            image = image * np.array(config.STD) + np.array(config.MEAN)
            image = np.clip(image, 0, 1)

            # -----------------------------
            # Calculate Dice Score
            # -----------------------------
            intersection = np.sum(prediction * mask)
            union = np.sum(prediction) + np.sum(mask)
            dice = (2 * intersection + 1e-8) / (union + 1e-8)

            # -----------------------------
            # Create Figure
            # -----------------------------
            plt.figure(figsize=(12, 4))

            # Original Image
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis("off")

            # Ground Truth
            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap="gray")
            plt.title("Ground Truth")
            plt.axis("off")

            # Prediction
            plt.subplot(1, 3, 3)
            plt.imshow(prediction, cmap="gray")
            plt.title(f"Prediction\nDice = {dice:.4f}")
            plt.axis("off")

            plt.tight_layout()

            # -----------------------------
            # Save Figure
            # -----------------------------
            plt.savefig(
                os.path.join(
                    config.COMPARISON_DIR,
                    f"prediction_{idx+1}.png"
                )
            )
            plt.close()

    model.train()

#=========================================
#Datasets
#=========================================
train_dataset = CarvanaDataset(
    image_dir=config.TRAIN_IMG_DIR,
    mask_dir=config.TRAIN_MASK_DIR,
    transform=train_transform,
)

val_dataset = CarvanaDataset(
    image_dir=config.VAL_IMG_DIR,
    mask_dir=config.VAL_MASK_DIR,
    transform=val_transform,
)

#=========================================
#DataLoaders
#=========================================
train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY,
)

#=========================================
#Model
#=========================================
model = UNET(
    in_channels=config.IN_CHANNELS,
    out_channels=config.OUT_CHANNELS,
    features=config.FEATURES,
).to(config.DEVICE)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE,
)

# ===================================================
# Training Loop
# ===================================================
best_dice= 0.0

for epoch in range(config.NUM_EPOCHS):
    print("=" * 60)
    print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
    print("=" * 60)

    train_loss = train_one_epoch(
        train_loader,
        model,
        optimizer,
        loss_fn,
        config.DEVICE,
    )
    
    val_loss, dice, iou = validate(
        val_loader,
        model,
        loss_fn,
        config.DEVICE,
    )

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["dice"].append(dice)
    history["iou"].append(iou)

    print(f"Train Loss      : {train_loss:.4f}")
    print(f"Validation Loss : {val_loss:.4f}")
    print(f"Dice Score      : {dice:.4f}")
    print(f"IoU Score       : {iou:.4f}")

    if dice > best_dice:
        best_dice = dice
        torch.save(
            model.state_dict(),
            config.CHECKPOINT_PATH,
        )

        print("Best model saved!")

model.load_state_dict(
    torch.load(
        config.CHECKPOINT_PATH,
        map_location=config.DEVICE,
    )
)
print("Loading best model checkpoint...")
plot_history(history)

save_predictions(val_loader, model, config.DEVICE)

print("\nTraining Completed!")