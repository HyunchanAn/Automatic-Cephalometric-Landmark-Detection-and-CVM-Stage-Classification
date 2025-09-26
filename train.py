import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from collections import OrderedDict

from dataset import HeatmapDataset
from model import UNetHeatmapModel # Use the new UNet model
from config import (
    DATASET_PATH, 
    IMAGE_SIZE, 
    CHECKPOINT_PATH, 
    NUM_LANDMARKS,
    NUM_WORKERS,
    PIN_MEMORY,
    VALID_BATCH_SIZE
)

# --- Hyperparameters ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
BATCH_SIZE = 4 # Keep small for 512x512 images
EPOCHS = 150
HEATMAP_OUTPUT_SIZE = (128, 128) # Corresponds to 512x512 input
HEATMAP_SIGMA = 2 # Best sigma from previous experiments
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.2
EARLY_STOP_PATIENCE = 20

# --- Transfer Learning Settings ---
# Start from our best 256px model to leverage its learned weights
PRETRAINED_MODEL_PATH = 'model_heatmap_resnet50_finetuned_mre4.5.pth'
FINE_TUNE_EPOCH = 15 # Unfreeze backbone after 15 epochs
FINE_TUNE_LR = 1e-5

def get_coords_from_heatmaps(heatmaps, image_size, heatmap_size):
    """Convert a batch of heatmaps to landmark coordinates."""
    batch_size, num_landmarks, h, w = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape(batch_size, num_landmarks, -1)
    max_indices = torch.argmax(heatmaps_reshaped, dim=2)
    y_coords = max_indices // w
    x_coords = max_indices % w
    coords = torch.stack([x_coords, y_coords], dim=2).float()
    scale_x = image_size[1] / w
    scale_y = image_size[0] / h
    coords[:, :, 0] *= scale_x
    coords[:, :, 1] *= scale_y
    return coords

def main():
    # --- 1. Setup Logging ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file_path = f'{timestamp}_unet_transfer_512px_log.csv' # New log file name
    with open(log_file_path, 'w') as f:
        f.write('# Model: UNetHeatmapModel (512px, Transfer), Loss: MSELoss\n')
        f.write('epoch,train_loss,valid_mre,learning_rate\n')

    # --- 2. Load Dataset ---
    print(f"Loading dataset with image size {IMAGE_SIZE} and heatmap size {HEATMAP_OUTPUT_SIZE}...")
    train_dataset = HeatmapDataset(dataset_folder_path=DATASET_PATH, mode="TRAIN", image_size=IMAGE_SIZE, output_size=HEATMAP_OUTPUT_SIZE, sigma=HEATMAP_SIGMA)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    valid_dataset = HeatmapDataset(dataset_folder_path=DATASET_PATH, mode="VALID", image_size=IMAGE_SIZE, output_size=HEATMAP_OUTPUT_SIZE, sigma=HEATMAP_SIGMA)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # --- 3. Initialize Model, Loss, and Optimizer ---
    print(f"Initializing UNetHeatmapModel on {DEVICE}...")
    model = UNetHeatmapModel(num_landmarks=NUM_LANDMARKS).to(DEVICE)
    loss_fn = nn.MSELoss()

    # --- Transfer Learning Logic ---
    if os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"Loading backbone weights from: {PRETRAINED_MODEL_PATH}")
        pretrained_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
        model_dict = model.state_dict()
        
        # Create a new state dict for the UNet model
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            # The old model had backbone weights wrapped in 'backbone.N...'. 
            # The new UNet model has them as 'layerN...'. We need to map them.
            if k.startswith('backbone.'):
                # e.g., backbone.0.conv1.weight -> layer0.0.weight
                # This mapping is complex because sequential names don't match layer names.
                # A simpler approach is to load only matching keys if names were consistent.
                # For this specific case, we assume the ResNet part is consistent.
                pass # Manual mapping is too complex, let's try loading what matches.

        # Filter out unnecessary keys and load the rest
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"Transferred {len(pretrained_dict)} matching layers from pre-trained model.")
        model.freeze_backbone()
        print("Model backbone frozen for feature extraction.")
    else:
        print("No pre-trained model found. Starting from scratch.")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, min_lr=1e-8)

    # --- 4. Training Loop ---
    best_mre = float('inf')
    early_stop_counter = 0
    fine_tuning_activated = False
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    print("Starting UNet model training with transfer learning...")
    for epoch in range(EPOCHS):
        if not fine_tuning_activated and epoch >= FINE_TUNE_EPOCH:
            print("\n--- Starting fine-tuning phase ---")
            model.unfreeze_backbone()
            optimizer = torch.optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, min_lr=1e-8)
            fine_tuning_activated = True
            print(f"Model backbone unfrozen. Optimizer reset with new learning rate: {FINE_TUNE_LR}")

        model.train()
        train_loss = 0.0
        for images, heatmaps_true, _ in train_loader:
            images, heatmaps_true = images.to(DEVICE), heatmaps_true.to(DEVICE)
            optimizer.zero_grad()
            heatmaps_pred = model(images)
            loss = loss_fn(heatmaps_pred, heatmaps_true)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- 5. Validation ---
        model.eval()
        total_radial_error = 0.0
        with torch.no_grad():
            for images, _, landmarks_true in valid_loader:
                images, landmarks_true = images.to(DEVICE), landmarks_true.to(DEVICE)
                heatmaps_pred = model(images)
                coords_pred = get_coords_from_heatmaps(heatmaps_pred, IMAGE_SIZE, HEATMAP_OUTPUT_SIZE)
                radial_error = torch.sqrt(((coords_pred - landmarks_true) ** 2).sum(dim=2))
                total_radial_error += radial_error.mean(dim=1).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        mre = total_radial_error / len(valid_dataset)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"--- Epoch [{epoch+1}/{EPOCHS}] ---")
        print(f"  Avg Train Loss: {avg_train_loss:.6f}")
        print(f"  Validation MRE (px): {mre:.4f}")
        print(f"  Current LR: {current_lr}")

        # --- 6. Log to CSV ---
        with open(log_file_path, 'a') as f:
            f.write(f'{epoch+1},{avg_train_loss:.6f},{mre:.4f},{current_lr}\n')

        # --- 7. Save Best Model & Early Stopping ---
        if mre < best_mre:
            best_mre = mre
            early_stop_counter = 0
            best_model_save_path = os.path.join(CHECKPOINT_PATH, 'best_unet_transfer_model_512px.pth')
            torch.save(model.state_dict(), best_model_save_path)
            print(f"  >>> New best model saved with MRE: {best_mre:.4f}")
        else:
            early_stop_counter += 1
            print(f"  (No improvement in MRE for {early_stop_counter}/{EARLY_STOP_PATIENCE} epochs)")

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after {EARLY_STOP_PATIENCE} epochs with no improvement.")
            break
        
        scheduler.step(mre)

    print("Finished Training.")

if __name__ == '__main__':
    main()