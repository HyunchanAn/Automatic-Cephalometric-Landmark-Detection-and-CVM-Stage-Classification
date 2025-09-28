
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, cohen_kappa_score, mean_absolute_error
import numpy as np
import os
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset_mil import MILDataset
from model_mil import AttentionMIL
import config

# --- CORAL Ordinal Loss --- #
def coral_loss(logits, levels, num_classes=6):
    """Computes the CORAL loss for ordinal regression in a vectorized way."""
    # levels: (batch_size), e.g., [2, 0, 1, 5, ...]
    # logits: (batch_size, num_classes - 1)
    
    # Create ordinal labels efficiently
    # if level is y (0-indexed), tasks 0 to y-1 are positive.
    levels = levels.long()
    ordinal_labels = torch.arange(num_classes - 1, device=logits.device).expand(len(levels), -1) < levels.unsqueeze(1)
    ordinal_labels = ordinal_labels.float()

    return nn.BCEWithLogitsLoss(reduction='mean')(logits, ordinal_labels)

def logits_to_prediction(logits):
    """Converts CORAL logits to predicted class levels."""
    # Create predictions by summing the number of positive logits
    return torch.sum(torch.sigmoid(logits) > 0.5, dim=1)

# --- Main Training Script --- #
def main():
    # Hyperparameters
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = config.BATCH_SIZE
    EPOCHS = 200 # As per memo
    LR = 1e-4 # Start with a lower LR for fine-tuning based approaches
    NUM_PATCHES = 16
    PATCH_SIZE = 128
    NUM_CLASSES = 6

    print(f"Using device: {DEVICE}")

    # --- 1. Setup Logging ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_filename = f"{timestamp}_mil_coral_log.csv"
    log_filepath = os.path.join("training_log", log_filename)
    os.makedirs("training_log", exist_ok=True)
    
    with open(log_filepath, 'w') as f:
        f.write('# Plan 1: AttentionMIL with CORAL Loss\n')
        f.write('epoch,train_loss,valid_loss,f1_macro,kappa,mae,learning_rate\n')
    print(f"Logging to {log_filepath}")

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Use project's convention
    ])

    # Datasets and DataLoaders
    train_dataset = MILDataset(mode='TRAIN', transform=transform, num_patches=NUM_PATCHES, patch_size=PATCH_SIZE)
    valid_dataset = MILDataset(mode='VALID', transform=transform, num_patches=NUM_PATCHES, patch_size=PATCH_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    # Model, Optimizer, Loss
    model = AttentionMIL(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=10, min_lr=1e-8) # Monitor kappa

    best_kappa = -1
    
    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for patches, levels in train_loader:
            patches, levels = patches.to(DEVICE), levels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(patches)
            loss = coral_loss(logits, levels, num_classes=NUM_CLASSES)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validation Loop ---
        model.eval()
        valid_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for patches, levels in valid_loader:
                patches, levels = patches.to(DEVICE), levels.to(DEVICE)
                logits = model(patches)
                loss = coral_loss(logits, levels, num_classes=NUM_CLASSES)
                valid_loss += loss.item()

                preds = logits_to_prediction(logits)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(levels.cpu().numpy())

        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        f1 = f1_score(all_labels, all_preds, average='macro')
        kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        mae = mean_absolute_error(all_labels, all_preds)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f} | F1: {f1:.4f} | Kappa: {kappa:.4f} | MAE: {mae:.4f}")

        # --- Log to CSV ---
        with open(log_filepath, 'a') as f:
            f.write(f'{epoch+1},{avg_train_loss:.6f},{avg_valid_loss:.6f},{f1:.4f},{kappa:.4f},{mae:.4f},{current_lr}\n')

        # Save best model
        if kappa > best_kappa:
            best_kappa = kappa
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_PATH, 'best_mil_model.pth'))
            print(f"** New best model saved with Kappa: {kappa:.4f} at epoch {epoch+1} **")
            
        scheduler.step(kappa) # Step scheduler on kappa score

if __name__ == '__main__':
    # Ensure checkpoint directory exists
    if not os.path.exists(config.CHECKPOINT_PATH):
        os.makedirs(config.CHECKPOINT_PATH)
    main()
