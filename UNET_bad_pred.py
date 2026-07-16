import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader

import UNET_config as config
from UNET_model import UNET
from UNET_dataset import CarvanaDataset

# ===========================================
# Transform
# ===========================================
val_transform = A.Compose([
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
])

# ===========================================
# Dataset
# ===========================================
val_dataset = CarvanaDataset(
    image_dir=config.VAL_IMG_DIR,
    mask_dir=config.VAL_MASK_DIR,
    transform=val_transform,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY,
)

# ===========================================
# Model
# ===========================================
model = UNET(
    in_channels=config.IN_CHANNELS,
    out_channels=config.OUT_CHANNELS,
    features=config.FEATURES,
).to(config.DEVICE)

model.load_state_dict(
    torch.load(
        config.CHECKPOINT_PATH,
        map_location=config.DEVICE,
    )
)


model.eval()
os.makedirs("results/worst_predictions", exist_ok=True)
samples = []
print("Evaluating validation set...\n")

with torch.no_grad():
    for idx, (images, masks) in enumerate(val_loader):
        images = images.to(config.DEVICE)
        masks = masks.unsqueeze(1).float().to(config.DEVICE)

        outputs = model(images)

        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()
        pred = preds[0].cpu().numpy().squeeze()
        gt = masks[0].cpu().numpy().squeeze()

        intersection = np.sum(pred * gt)
        union = np.sum(pred) + np.sum(gt)

        dice = (2 * intersection + 1e-8) / (union + 1e-8)

        image = images[0].cpu().permute(1, 2, 0).numpy()
        image = image * np.array(config.STD) + np.array(config.MEAN)
        image = np.clip(image, 0, 1)

        samples.append({
            "dice": dice,
            "image": image,
            "gt": gt,
            "pred": pred,
            "index": idx,
        })

# ===========================================
# Sort by Dice
# ===========================================
samples = sorted(samples, key=lambda x: x["dice"])
print("Worst Dice Scores")

for s in samples[:5]:
    print(f"Image {s['index']}  Dice = {s['dice']:.4f}")

# ===========================================
# Save worst predictions
# ===========================================
for rank, sample in enumerate(samples[:5], start=1):
    image = sample["image"]
    gt = sample["gt"]
    pred = sample["pred"]
    dice = sample["dice"]

    error = np.zeros((gt.shape[0], gt.shape[1], 3))

    # False Positive = RED
    error[(pred == 1) & (gt == 0)] = [1, 0, 0]

    # False Negative = BLUE
    error[(pred == 0) & (gt == 1)] = [0, 0, 1]

    plt.figure(figsize=(16,4))

    plt.subplot(1,4,1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(gt,cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(pred,cmap="gray")
    plt.title(f"Prediction\nDice={dice:.4f}")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.imshow(error)
    plt.title("Error Map")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        f"results/worst_predictions/worst_{rank}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ===========================================
# Save CSV
# ===========================================
with open("results/worst_predictions/worst_scores.csv","w") as f:
    f.write("Rank,Validation Index,Dice\n")
    for rank, sample in enumerate(samples[:5], start=1):
        f.write(f"{rank},{sample['index']},{sample['dice']:.6f}\n")

print("\nDone!")