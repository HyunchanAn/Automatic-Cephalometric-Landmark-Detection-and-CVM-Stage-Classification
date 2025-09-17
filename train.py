import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import AarizDataset
from model import CephNet

# --- Hyperparameters ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.001
BATCH_SIZE = 16 # Adjust based on your system's memory
EPOCHS = 10 # Start with a small number of epochs
IMAGE_SIZE = (256, 256) # The model expects a fixed size
DATASET_PATH = "C:\\Users\\user\\Documents\\Github\\A-Benchmark-Dataset-for-Automatic-Cephalometric-Landmark-Detection-and-CVM-Stage-Classification\\Aariz"


def main():
    # --- 1. Load Dataset ---
    print("Loading dataset...")
    
    # Define transformations for the training set with data augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(), # 50% chance to flip the image horizontally
        transforms.RandomRotation(10), # Rotate the image by a random angle up to 10 degrees
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define transformations for the validation set (no augmentation)
    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = AarizDataset(dataset_folder_path=DATASET_PATH, mode="TRAIN", transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    valid_dataset = AarizDataset(dataset_folder_path=DATASET_PATH, mode="VALID", transform=valid_transform)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 2. Initialize Model, Loss, and Optimizer ---
    print(f"Initializing model on {DEVICE}...")
    model = CephNet().to(DEVICE)

    landmark_loss_fn = nn.MSELoss()
    cvm_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Training Loop ---
    print("Starting training with data augmentation...")
    for epoch in range(EPOCHS):
        model.train() # Set model to training mode
        train_loss = 0.0
        for batch_idx, (images, landmarks_true, cvm_stages_true) in enumerate(train_loader):
            images, landmarks_true, cvm_stages_true = images.to(DEVICE), landmarks_true.to(DEVICE), cvm_stages_true.to(DEVICE)
            optimizer.zero_grad()
            landmarks_pred, cvm_stages_pred = model(images)
            loss_landmarks = landmark_loss_fn(landmarks_pred, landmarks_true)
            loss_cvm = cvm_loss_fn(cvm_stages_pred, cvm_stages_true)
            total_loss = loss_landmarks + loss_cvm # You can also weight the losses
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        # --- 4. Validation ---
        model.eval() # Set model to evaluation mode
        valid_loss = 0.0
        with torch.no_grad():
            for images, landmarks_true, cvm_stages_true in valid_loader:
                images, landmarks_true, cvm_stages_true = images.to(DEVICE), landmarks_true.to(DEVICE), cvm_stages_true.to(DEVICE)
                landmarks_pred, cvm_stages_pred = model(images)
                loss_landmarks = landmark_loss_fn(landmarks_pred, landmarks_true)
                loss_cvm = cvm_loss_fn(cvm_stages_pred, cvm_stages_true)
                total_loss = loss_landmarks + loss_cvm
                valid_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}")

    print("Finished Training.")

    # --- 5. Save the Model ---
    # After training, you can save the model's state dictionary
    # torch.save(model.state_dict(), 'cephnet_model_augmented.pth')
    # print("Model saved to cephnet_model_augmented.pth")

if __name__ == '__main__':
    main()