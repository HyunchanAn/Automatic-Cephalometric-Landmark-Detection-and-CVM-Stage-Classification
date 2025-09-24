import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from datetime import datetime # Import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset_roi_cvm import RoiCvmDataset # Import the new CVM-only dataset
from model_cvm_only import CvmOnlyNet # Import the new CVM-only model
from focal_loss import FocalLoss # Import FocalLoss
from config import (
    CHECKPOINT_PATH, ROI_DATASET_PATH, IMAGE_SIZE_CVM as IMAGE_SIZE, # Use 224x224 image size
    NUM_WORKERS, PIN_MEMORY, VALID_BATCH_SIZE
)

# Set number of threads for CPU operations
torch.set_num_threads(1)

# --- Hyperparameters ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
EPOCHS = 100
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.1
EARLY_STOP_PATIENCE = 30

def main():
    # --- 0. Setup Logging ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file_path = f'{timestamp}_cvm_only_training_log.csv'
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as f:
            f.write(f'# Model: CvmOnlyNet (Backbone: ResNet-50), Loss: FocalLoss (gamma=2.0, weighted)\n')
            f.write('epoch,train_loss,valid_loss,cvm_accuracy,cvm_f1\n')

    # --- 1. Load Dataset ---
    print(f"Loading dataset with image size {IMAGE_SIZE}...")
    train_dataset = RoiCvmDataset(dataset_folder_path=ROI_DATASET_PATH, mode="TRAIN", image_size=IMAGE_SIZE)
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    valid_dataset = RoiCvmDataset(dataset_folder_path=ROI_DATASET_PATH, mode="VALID", image_size=IMAGE_SIZE)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    # --- 2. Initialize Model, Loss, and Optimizer ---
    print(f"Initializing CvmOnlyNet model on {DEVICE}...")
    model = CvmOnlyNet().to(DEVICE)

    # Define class weights to counteract class imbalance
    class_weights = torch.tensor([6.6667, 3.3333, 3.4568, 0.6813, 0.3733, 0.8974], dtype=torch.float).to(DEVICE)
    cvm_loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, min_lr=1e-6)

    # --- 3. Load Checkpoint if exists ---
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = -1

    # Checkpoint logic can be added here if needed, simplified for now

    # --- 4. Training Loop ---
    # Create checkpoint directory if it does not exist
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
        
    early_stop_counter = 0
    print("Starting CVM-only training...")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_idx, (images, cvm_stages_true) in enumerate(train_loader):
            images, cvm_stages_true = images.to(DEVICE), cvm_stages_true.to(DEVICE)
            
            optimizer.zero_grad()
            cvm_stages_pred = model(images)
            loss = cvm_loss_fn(cvm_stages_pred, cvm_stages_true)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- 5. Validation and Metrics ---
        model.eval()
        valid_loss = 0.0
        all_cvm_preds = []
        all_cvm_true = []
        with torch.no_grad():
            for images, cvm_stages_true in valid_loader:
                images, cvm_stages_true = images.to(DEVICE), cvm_stages_true.to(DEVICE)
                
                cvm_stages_pred = model(images)
                loss = cvm_loss_fn(cvm_stages_pred, cvm_stages_true)
                valid_loss += loss.item()

                cvm_preds = cvm_stages_pred.argmax(dim=1)
                all_cvm_preds.append(cvm_preds.cpu().numpy())
                all_cvm_true.append(cvm_stages_true.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        
        all_cvm_preds = np.concatenate(all_cvm_preds)
        all_cvm_true = np.concatenate(all_cvm_true)
        
        cvm_accuracy = accuracy_score(all_cvm_true, all_cvm_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_cvm_true, all_cvm_preds, average='macro', zero_division=0)

        print(f"--- Epoch [{epoch+1}/{EPOCHS}] ---")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Avg Valid Loss: {avg_valid_loss:.4f}")
        print(f"  CVM Accuracy: {cvm_accuracy:.4f}")
        print(f"  CVM F1-Score: {f1:.4f}")

        # --- 7. Log to CSV ---
        with open(log_file_path, 'a') as f:
            f.write(f'{epoch+1},{avg_train_loss:.4f},{avg_valid_loss:.4f},{cvm_accuracy:.4f},{f1:.4f}\n')

        # --- 6. Save Best Model & Early Stopping (based on validation loss) ---
        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            best_epoch = epoch + 1
            early_stop_counter = 0
            best_model_save_path = os.path.join(CHECKPOINT_PATH, 'best_cvm_model.pth')
            torch.save(model.state_dict(), best_model_save_path)
            print(f"  >>> New best model saved with Valid Loss: {best_val_loss:.4f} at epoch {best_epoch}")
        else:
            early_stop_counter += 1
            print(f"  (No improvement in Valid Loss for {early_stop_counter}/{EARLY_STOP_PATIENCE} epochs)")

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after {EARLY_STOP_PATIENCE} epochs with no improvement.")
            break
        
        scheduler.step(avg_valid_loss)

    print("Finished CVM-only training.")

if __name__ == '__main__':
    main()
