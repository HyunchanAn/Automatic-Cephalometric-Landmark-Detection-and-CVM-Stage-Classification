import torch
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torchvision import transforms
from torch.utils.data import DataLoader

from model import AdvancedCephNet
from dataset import AarizDataset
from config import CHECKPOINT_PATH, NUM_LANDMARKS, DATASET_PATH, IMAGE_SIZE

def main():
    # --- 1. Setup ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # --- 2. Load Model ---
    print("Loading best model...")
    model = AdvancedCephNet().to(DEVICE)

    best_model_path = os.path.join(CHECKPOINT_PATH, 'best_model.pth')
    if not os.path.exists(best_model_path):
        print(f"Error: Best model not found at {best_model_path}. Please train the model first.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval() # Set model to evaluation mode
    print("Best model loaded successfully.")

    # --- 3. Load Test Dataset ---
    print("Loading test dataset...")
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = AarizDataset(dataset_folder_path=DATASET_PATH, mode="TEST", transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False) # Use batch_size=1 for evaluation

    if len(test_dataset) == 0:
        print("Test dataset is empty. Cannot perform evaluation.")
        return

    # --- 4. Evaluate on Test Set ---
    print("Evaluating on test set...")
    total_radial_error = 0.0
    all_cvm_preds = []
    all_cvm_true = []

    with torch.no_grad():
        for images, landmarks_true, cvm_stages_true in test_loader:
            images, landmarks_true, cvm_stages_true = images.to(DEVICE), landmarks_true.to(DEVICE), cvm_stages_true.to(DEVICE)
            
            landmarks_pred, cvm_stages_pred = model(images)
            
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

    # Calculate overall metrics
    mre = total_radial_error / len(test_dataset)
    
    all_cvm_preds = np.concatenate(all_cvm_preds)
    all_cvm_true = np.concatenate(all_cvm_true)
    
    cvm_accuracy = accuracy_score(all_cvm_true, all_cvm_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_cvm_true, all_cvm_preds, average='macro', zero_division=0)

    print("\n--- Test Set Evaluation Results ---")
    print(f"  Landmark MRE (px): {mre:.4f}")
    print(f"  CVM Accuracy: {cvm_accuracy:.4f}")
    print(f"  CVM F1-Score: {f1:.4f}")
    print("-----------------------------------")

if __name__ == '__main__':
    main()
