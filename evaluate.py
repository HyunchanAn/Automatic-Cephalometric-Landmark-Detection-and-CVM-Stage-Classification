import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import albumentations as A

# Use the model and dataset compatible with our best performing model
from model import HeatmapModel # Reverted to the correct model class for the saved weights
from dataset import HeatmapDataset 
from config import DATASET_PATH, NUM_LANDMARKS

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Settings for our best model (MRE 4.5px) ---
MODEL_PATH = 'models/model_heatmap_resnet50_finetuned_mre4.5.pth'
IMAGE_SIZE = (256, 256)
HEATMAP_OUTPUT_SIZE = (64, 64) # 256 / 4 = 64
BATCH_SIZE = 16

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

def evaluate(model, loader):
    """Evaluates the model on the test set and returns the MRE."""
    model.eval()
    total_radial_error = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, _, landmarks_true in tqdm(loader, desc="Evaluating Test Set"):
            images, landmarks_true = images.to(DEVICE), landmarks_true.to(DEVICE)
            
            heatmaps_pred = model(images)
            coords_pred = get_coords_from_heatmaps(heatmaps_pred, IMAGE_SIZE, HEATMAP_OUTPUT_SIZE)
            
            radial_error = torch.sqrt(((coords_pred - landmarks_true) ** 2).sum(dim=2))
            total_radial_error += radial_error.sum().item()
            num_samples += images.size(0) * NUM_LANDMARKS

    mre = total_radial_error / num_samples
    return mre

def visualize(model, dataset, index):
    """Visualizes model prediction on a single image."""
    print(f"Visualizing prediction for test image at index {index}...")
    model.eval()

    # Get the specific data sample
    image_tensor, _, landmarks_true = dataset[index]
    image_path = os.path.join(DATASET_PATH, 'test', 'Cephalograms', dataset.images_list[index])
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

    with torch.no_grad():
        heatmaps_pred = model(image_tensor.unsqueeze(0).to(DEVICE))
        coords_pred = get_coords_from_heatmaps(heatmaps_pred, IMAGE_SIZE, HEATMAP_OUTPUT_SIZE).cpu().numpy()[0]

    landmarks_true = landmarks_true.cpu().numpy()

    # Draw true landmarks (green) and predicted landmarks (red)
    for i in range(NUM_LANDMARKS):
        true_x, true_y = int(landmarks_true[i, 0]), int(landmarks_true[i, 1])
        pred_x, pred_y = int(coords_pred[i, 0]), int(coords_pred[i, 1])

        cv2.circle(original_image, (true_x, true_y), radius=3, color=(0, 255, 0), thickness=-1) # Green for true
        cv2.circle(original_image, (pred_x, pred_y), radius=3, color=(0, 0, 255), thickness=-1) # Red for predicted
        cv2.line(original_image, (true_x, true_y), (pred_x, pred_y), color=(255, 255, 0), thickness=1)

    output_path = 'evaluation_visualization.jpg'
    cv2.imwrite(output_path, original_image)
    print(f"Visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate the landmark detection model.')
    parser.add_argument('--visualize-index', type=int, help='Index of the test image to visualize.')
    args = parser.parse_args()

    # --- Load Model ---
    print(f"Loading best model from {MODEL_PATH}...")
    model = HeatmapModel(num_landmarks=NUM_LANDMARKS).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}.")
        return

    # Load the state dict directly, as the class now matches the saved file.
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Model loaded successfully.")
    model.eval()

    # --- Load Test Dataset ---
    test_dataset = HeatmapDataset(
        dataset_folder_path=DATASET_PATH, 
        mode="TEST", 
        image_size=IMAGE_SIZE, # Explicitly pass the correct image size
        output_size=HEATMAP_OUTPUT_SIZE
    )

    if args.visualize_index is not None:
        if 0 <= args.visualize_index < len(test_dataset):
            visualize(model, test_dataset, args.visualize_index)
        else:
            print(f"Error: visualize_index must be between 0 and {len(test_dataset) - 1}")
    else:
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        mre = evaluate(model, test_loader)
        print("\n--- Test Set Evaluation Results ---")
        print(f"  Landmark MRE (px): {mre:.4f}")
        print("-----------------------------------")

if __name__ == '__main__':
    main()