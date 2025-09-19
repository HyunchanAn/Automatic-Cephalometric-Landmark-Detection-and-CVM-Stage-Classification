import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import AarizDataset
from model import AdvancedCephNet
from config import CHECKPOINT_PATH, NUM_LANDMARKS, DATASET_PATH, IMAGE_SIZE, VALID_BATCH_SIZE, NUM_WORKERS, PIN_MEMORY

# --- Hyperparameters ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.001
BATCH_SIZE = 8 # Adjust based on your system's memory
EPOCHS = 200 # Number of epochs for training
LR_SCHEDULER_PATIENCE = 5 # Number of epochs with no improvement after which learning rate will be reduced
LR_SCHEDULER_FACTOR = 0.1 # Factor by which the learning rate will be reduced


def main():
    # --- 1. Load Dataset ---
    print("Loading dataset...")
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = AarizDataset(dataset_folder_path=DATASET_PATH, mode="TRAIN", transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    valid_dataset = AarizDataset(dataset_folder_path=DATASET_PATH, mode="VALID", transform=valid_transform)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # --- 2. Initialize Model, Loss, and Optimizer ---
    print(f"Initializing model on {DEVICE}...")
    model = AdvancedCephNet().to(DEVICE)

    landmark_loss_fn = nn.MSELoss()
    cvm_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, min_lr=1e-6)

    # --- 3. Load Checkpoint if exists ---
    start_epoch = 0
    best_mre = float('inf')
    best_epoch = -1

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint_list = os.listdir(CHECKPOINT_PATH)
        if checkpoint_list:
            latest_checkpoint_file = max([f for f in checkpoint_list if f.startswith('checkpoint_epoch_')], 
                                         key=lambda f: int(f.split('_')[-1].split('.')[0]))
            checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, latest_checkpoint_file))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if 'best_mre' in checkpoint: # Check if best_mre was saved in checkpoint
                best_mre = checkpoint['best_mre']
                best_epoch = checkpoint['best_epoch']
            print(f"Resuming training from epoch {start_epoch} with best MRE {best_mre:.4f} from epoch {best_epoch}")

    # --- 4. Training Loop ---
    print("Starting training...")
    FINE_TUNE_EPOCH = 20 # Epoch to start fine-tuning (unfreezing backbone)
    FINE_TUNE_LR_FACTOR = 0.1 # Factor to reduce LR for fine-tuning

    for epoch in range(start_epoch, EPOCHS):
        if epoch == FINE_TUNE_EPOCH:
            print(f"Unfreezing ResNet backbone and reducing LR at epoch {epoch}...")
            model.unfreeze()
            # Re-initialize optimizer with a lower learning rate for fine-tuning
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * FINE_TUNE_LR_FACTOR)
            # Re-initialize scheduler as well, if needed, or adjust its parameters
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, min_lr=1e-7)

        model.train()
        train_loss = 0.0
        for batch_idx, (images, landmarks_true, cvm_stages_true) in enumerate(train_loader):
            images, landmarks_true, cvm_stages_true = images.to(DEVICE), landmarks_true.to(DEVICE), cvm_stages_true.to(DEVICE)
            optimizer.zero_grad()
            landmarks_pred, cvm_stages_pred = model(images)
            loss_landmarks = landmark_loss_fn(landmarks_pred, landmarks_true)
            loss_cvm = cvm_loss_fn(cvm_stages_pred, cvm_stages_true)
            total_loss = loss_landmarks + loss_cvm
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        # --- 5. Validation and Metrics ---
        model.eval()
        valid_loss = 0.0
        total_radial_error = 0.0
        all_cvm_preds = []
        all_cvm_true = []
        with torch.no_grad():
            for images, landmarks_true, cvm_stages_true in valid_loader:
                images, landmarks_true, cvm_stages_true = images.to(DEVICE), landmarks_true.to(DEVICE), cvm_stages_true.to(DEVICE)
                
                landmarks_pred, cvm_stages_pred = model(images)
                
                # Calculate validation loss
                loss_landmarks = landmark_loss_fn(landmarks_pred, landmarks_true)
                loss_cvm = cvm_loss_fn(cvm_stages_pred, cvm_stages_true)
                total_loss = loss_landmarks + loss_cvm
                valid_loss += total_loss.item()

                # 1. Landmark Mean Radial Error (MRE)
                landmarks_pred = landmarks_pred.view(-1, NUM_LANDMARKS, 2)
                landmarks_true = landmarks_true.view(-1, NUM_LANDMARKS, 2)
                radial_error = torch.sqrt(((landmarks_pred - landmarks_true) ** 2).sum(dim=2))
                total_radial_error += radial_error.mean(dim=1).sum().item()

                # 2. CVM Classification Metrics
                cvm_preds = cvm_stages_pred.argmax(dim=1)
                cvm_true = cvm_stages_true.argmax(dim=1)
                all_cvm_preds.append(cvm_preds.cpu().numpy())
                all_cvm_true.append(cvm_true.cpu().numpy())

        # Calculate average metrics for the epoch
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        mre = total_radial_error / len(valid_dataset)
        
        all_cvm_preds = np.concatenate(all_cvm_preds)
        all_cvm_true = np.concatenate(all_cvm_true)
        
        cvm_accuracy = accuracy_score(all_cvm_true, all_cvm_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_cvm_true, all_cvm_preds, average='macro', zero_division=0)

        print(f"--- Epoch [{epoch+1}/{EPOCHS}] ---")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Avg Valid Loss: {avg_valid_loss:.4f}")
        print(f"  Landmark MRE (px): {mre:.4f}")
        print(f"  CVM Accuracy: {cvm_accuracy:.4f}")
        print(f"  CVM F1-Score: {f1:.4f}")

        # --- 6. Save Checkpoint and Best Model ---
        if not os.path.exists(CHECKPOINT_PATH):
            os.makedirs(CHECKPOINT_PATH)
        
        # Save current checkpoint
        checkpoint_save_path = os.path.join(CHECKPOINT_PATH, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mre': best_mre,
            'best_epoch': best_epoch,
        }, checkpoint_save_path)

        # Save best model if current MRE is better
        if mre < best_mre:
            best_mre = mre
            best_epoch = epoch + 1
            best_model_save_path = os.path.join(CHECKPOINT_PATH, 'best_model.pth')
            torch.save(model.state_dict(), best_model_save_path)
            print(f"  >>> New best model saved with MRE: {best_mre:.4f} at epoch {best_epoch}")
        
        # Step the learning rate scheduler
        scheduler.step(mre)

    print("Finished Training.")

if __name__ == '__main__':
    main()