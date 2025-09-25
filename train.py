import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime

from dataset import HeatmapDataset
from model import HeatmapModel
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
BATCH_SIZE = 8
EPOCHS = 100
HEATMAP_OUTPUT_SIZE = (64, 64)
HEATMAP_SIGMA = 2
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.2
EARLY_STOP_PATIENCE = 15

# Fine-tuning settings
PRETRAINED_MODEL_PATH = os.path.join(CHECKPOINT_PATH, 'heatmap_resnet50_frozen_mre38.4.pth')
FINE_TUNE_EPOCH = 20
FINE_TUNE_LR = 1e-5

def get_coords_from_heatmaps(heatmaps):
    """Convert a batch of heatmaps to landmark coordinates."""
    batch_size, num_landmarks, h, w = heatmaps.shape
    
    heatmaps_reshaped = heatmaps.reshape(batch_size, num_landmarks, -1)
    max_indices = torch.argmax(heatmaps_reshaped, dim=2)
    
    y_coords = max_indices // w
    x_coords = max_indices % w
    
    coords = torch.stack([x_coords, y_coords], dim=2).float()
    
    scale_x = IMAGE_SIZE[1] / w
    scale_y = IMAGE_SIZE[0] / h
    coords[:, :, 0] *= scale_x
    coords[:, :, 1] *= scale_y
    
    return coords

def main():
    # --- 1. Setup Logging ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file_path = f'{timestamp}_heatmap_finetune_log.csv'
    with open(log_file_path, 'w') as f:
        f.write('# Model: HeatmapModel (Backbone: ResNet-50), Loss: MSELoss, Fine-tuned\n')
        f.write('epoch,train_loss,valid_mre,learning_rate\n')

    # --- 2. Load Dataset ---
    print(f"Loading dataset with image size {IMAGE_SIZE} and heatmap size {HEATMAP_OUTPUT_SIZE}...")
    train_dataset = HeatmapDataset(dataset_folder_path=DATASET_PATH, mode="TRAIN", output_size=HEATMAP_OUTPUT_SIZE, sigma=HEATMAP_SIGMA)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    valid_dataset = HeatmapDataset(dataset_folder_path=DATASET_PATH, mode="VALID", output_size=HEATMAP_OUTPUT_SIZE, sigma=HEATMAP_SIGMA)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # --- 3. Initialize Model, Loss, and Optimizer ---
    print(f"Initializing HeatmapModel on {DEVICE}...")
    model = HeatmapModel(num_landmarks=NUM_LANDMARKS).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, min_lr=1e-7)

    # Load weights from the feature extraction phase if available
    if os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"Loading pre-trained weights from {PRETRAINED_MODEL_PATH}")
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
    else:
        print("No pre-trained model found. Starting from scratch.")

    # --- 4. Training Loop ---
    best_mre = float('inf')
    early_stop_counter = 0
    fine_tuning_activated = False
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    print("Starting heatmap model training...")
    for epoch in range(EPOCHS):
        # --- Fine-tuning activation ---
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
                coords_pred = get_coords_from_heatmaps(heatmaps_pred)
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
            best_model_save_path = os.path.join(CHECKPOINT_PATH, 'best_heatmap_model_finetuned.pth')
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