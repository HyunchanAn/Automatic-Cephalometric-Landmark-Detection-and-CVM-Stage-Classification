import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from tqdm import tqdm

from dataset_roi_cvm import RoiCvmDataset
from model_cvm_unet import UNetCVMClassifier # Use the new UNet model for CVM
from focal_loss import FocalLoss
from config import (
    ROI_DATASET_PATH, 
    CHECKPOINT_PATH,
    NUM_WORKERS,
    PIN_MEMORY,
    VALID_BATCH_SIZE
)

# --- Hyperparameters ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EPOCHS = 100
IMAGE_SIZE_CVM = (224, 224) # Standard size for ImageNet models
LR_SCHEDULER_PATIENCE = 10
LR_SCHEDULER_FACTOR = 0.1
EARLY_STOP_PATIENCE = 25

def main():
    # --- 1. Setup Logging ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file_path = f'{timestamp}_unet_cvm_log.csv'
    with open(log_file_path, 'w') as f:
        f.write('# Model: UNetCVMClassifier, Loss: FocalLoss\n')
        f.write('epoch,train_loss,valid_loss,cvm_accuracy,cvm_f1\n')

    # --- 2. Load Dataset ---
    print(f"Loading ROI dataset with image size {IMAGE_SIZE_CVM}...")
    train_dataset = RoiCvmDataset(dataset_folder_path=ROI_DATASET_PATH, mode="TRAIN", image_size=IMAGE_SIZE_CVM)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    valid_dataset = RoiCvmDataset(dataset_folder_path=ROI_DATASET_PATH, mode="VALID", image_size=IMAGE_SIZE_CVM)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # --- 3. Initialize Model, Loss, and Optimizer ---
    print(f"Initializing UNetCVMClassifier on {DEVICE}...")
    model = UNetCVMClassifier().to(DEVICE)
    
    # Class weights from previous experiments to counteract class imbalance
    class_weights = torch.tensor([6.6667, 3.3333, 3.4568, 0.6813, 0.3733, 0.8974], dtype=torch.float).to(DEVICE)
    loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, min_lr=1e-7)

    # --- 4. Training Loop ---
    best_f1 = 0.0
    early_stop_counter = 0
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    print("Starting UNet CVM classifier training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, cvm_stages_true in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            images, cvm_stages_true = images.to(DEVICE), cvm_stages_true.to(DEVICE)
            
            optimizer.zero_grad()
            cvm_stages_pred = model(images)
            loss = loss_fn(cvm_stages_pred, cvm_stages_true)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- 5. Validation ---
        model.eval()
        valid_loss = 0.0
        all_cvm_preds = []
        all_cvm_true = []
        with torch.no_grad():
            for images, cvm_stages_true in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]"):
                images, cvm_stages_true = images.to(DEVICE), cvm_stages_true.to(DEVICE)
                cvm_stages_pred = model(images)
                loss = loss_fn(cvm_stages_pred, cvm_stages_true)
                valid_loss += loss.item()

                cvm_preds = cvm_stages_pred.argmax(dim=1)
                all_cvm_preds.append(cvm_preds.cpu().numpy())
                all_cvm_true.append(cvm_stages_true.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        
        all_cvm_preds = np.concatenate(all_cvm_preds)
        all_cvm_true = np.concatenate(all_cvm_true)
        
        cvm_accuracy = accuracy_score(all_cvm_true, all_cvm_preds)
        f1 = f1_score(all_cvm_true, all_cvm_preds, average='macro', zero_division=0)

        print(f"--- Epoch [{epoch+1}/{EPOCHS}] ---")
        print(f"  Avg Train Loss: {avg_train_loss:.4f} | Avg Valid Loss: {avg_valid_loss:.4f}")
        print(f"  CVM Accuracy: {cvm_accuracy:.4f} | CVM F1-Score: {f1:.4f}")

        with open(log_file_path, 'a') as f:
            f.write(f'{epoch+1},{avg_train_loss:.4f},{avg_valid_loss:.4f},{cvm_accuracy:.4f},{f1:.4f}\n')

        # Save best model based on F1-score
        if f1 > best_f1:
            best_f1 = f1
            early_stop_counter = 0
            best_model_save_path = os.path.join(CHECKPOINT_PATH, 'best_unet_cvm_model.pth')
            torch.save(model.state_dict(), best_model_save_path)
            print(f"  >>> New best model saved with F1-Score: {best_f1:.4f}")
        else:
            early_stop_counter += 1
            print(f"  (No improvement in F1-Score for {early_stop_counter}/{EARLY_STOP_PATIENCE} epochs)")

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after {EARLY_STOP_PATIENCE} epochs with no improvement.")
            break
        
        scheduler.step(f1)

    print("Finished CVM classifier training.")

if __name__ == '__main__':
    main()
