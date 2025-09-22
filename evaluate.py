import torch
import numpy as np
import os
import cv2
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from model import AdvancedCephNet
from dataset import AarizDataset
from config import CHECKPOINT_PATH, NUM_LANDMARKS, DATASET_PATH, IMAGE_SIZE, CVM_STAGES

def visualize_single_prediction(model, device, test_dataset, index):
    """
    Visualizes the model's prediction for a single image from the test set.
    """
    print(f"\n--- Generating visualization for image index {index} ---")

    if index >= len(test_dataset):
        print(f"Error: Index {index} is out of bounds for the test set (size: {len(test_dataset)}).")
        return

    # --- 1. Get Data ---
    # Get the transformed image for the model and the true labels
    image_for_model, landmarks_true_flat, cvm_stage_true_onehot = test_dataset[index]
    image_tensor = image_for_model.unsqueeze(0).to(device)

    # Get the original image for visualization
    image_name = test_dataset.images_list[index]
    image_path = os.path.join(test_dataset.images_root_path, image_name)
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read original image at {image_path}")
        return

    # --- 2. Run Inference ---
    print("Running inference for visualization...")
    with torch.no_grad():
        landmarks_pred_flat, cvm_stage_pred_logits = model(image_tensor)

    # --- 3. Process Predictions and Ground Truth ---
    landmarks_pred = landmarks_pred_flat.view(NUM_LANDMARKS, 2).cpu().numpy()
    landmarks_true = landmarks_true_flat.view(NUM_LANDMARKS, 2).cpu().numpy()
    
    cvm_pred_idx = cvm_stage_pred_logits.argmax(dim=1).item()
    cvm_true_idx = cvm_stage_true_onehot.argmax().item()

    cvm_pred_title = next((stage["title"] for stage in CVM_STAGES.values() if stage["value"] == cvm_pred_idx + 1), "Unknown")
    cvm_true_title = next((stage["title"] for stage in CVM_STAGES.values() if stage["value"] == cvm_true_idx + 1), "Unknown")

    # --- 4. Create Visualization ---
    print("Drawing landmarks on the image...")
    vis_image = original_image.copy()
    
    original_height, original_width, _ = original_image.shape
    model_height, model_width = IMAGE_SIZE

    scale_x = original_width / model_width
    scale_y = original_height / model_height

    # Draw landmarks
    for i in range(NUM_LANDMARKS):
        # True landmarks (Green)
        true_x = int(landmarks_true[i, 0] * scale_x)
        true_y = int(landmarks_true[i, 1] * scale_y)
        cv2.circle(vis_image, (true_x, true_y), radius=5, color=(0, 255, 0), thickness=-1)

        # Predicted landmarks (Red)
        pred_x = int(landmarks_pred[i, 0] * scale_x)
        pred_y = int(landmarks_pred[i, 1] * scale_y)
        cv2.circle(vis_image, (pred_x, pred_y), radius=5, color=(0, 0, 255), thickness=-1)
        
        # Line between true and predicted
        cv2.line(vis_image, (true_x, true_y), (pred_x, pred_y), color=(255, 255, 0), thickness=1)

    # Add text for CVM stages and legend
    cv2.putText(vis_image, f"True CVM: {cvm_true_title}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(vis_image, f"Pred CVM: {cvm_pred_title}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.circle(vis_image, (10, 110), 5, (0, 255, 0), -1)
    cv2.putText(vis_image, "True Landmark", (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.circle(vis_image, (10, 140), 5, (0, 0, 255), -1)
    cv2.putText(vis_image, "Predicted Landmark", (25, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Save the output image
    output_filename = "visualization_output.png"
    cv2.imwrite(output_filename, vis_image)
    print(f"Visualization saved to {output_filename}")

def main(args):
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
    model.eval()
    print("Best model loaded successfully.")

    # --- 3. Load Test Dataset ---
    print("Loading test dataset...")
    # Transforms are now handled by the dataset class itself
    test_dataset = AarizDataset(dataset_folder_path=DATASET_PATH, mode="TEST", image_size=IMAGE_SIZE)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    if len(test_dataset) == 0:
        print("Test dataset is empty. Cannot perform evaluation.")
        return

    # --- 4. Quantitative Evaluation on Test Set ---
    print("Evaluating on the entire test set...")
    total_radial_error = 0.0
    all_cvm_preds = []
    all_cvm_true = []

    with torch.no_grad():
        for images, landmarks_true, cvm_stages_true in test_loader:
            images, landmarks_true, cvm_stages_true = images.to(DEVICE), landmarks_true.to(DEVICE), cvm_stages_true.to(DEVICE)
            
            landmarks_pred, cvm_stages_pred = model(images)
            
            # Landmark Mean Radial Error (MRE)
            landmarks_pred_unflat = landmarks_pred.view(-1, NUM_LANDMARKS, 2)
            landmarks_true_unflat = landmarks_true.view(-1, NUM_LANDMARKS, 2)
            radial_error = torch.sqrt(((landmarks_pred_unflat - landmarks_true_unflat) ** 2).sum(dim=2))
            total_radial_error += radial_error.mean(dim=1).sum().item()

            # CVM Classification Metrics
            cvm_preds = cvm_stages_pred.argmax(dim=1)
            cvm_true = cvm_stages_true.argmax(dim=1)
            all_cvm_preds.append(cvm_preds.cpu().numpy())
            all_cvm_true.append(cvm_true.cpu().numpy())

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

    # --- 5. Optional Visualization ---
    if args.visualize_index is not None:
        visualize_single_prediction(model, DEVICE, test_dataset, args.visualize_index)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the model on the test set and optionally visualize a prediction.")
    parser.add_argument('--visualize-index', type=int, default=None, help='Index of a test set image to visualize. If provided, evaluation is run and a visualization is saved.')
    args = parser.parse_args()
    main(args)