import torch
import cv2
import numpy as np
import os
import json
from tqdm import tqdm

from model import AdvancedCephNet
from dataset import AarizDataset
from config import (
    DATASET_PATH, IMAGE_SIZE, NUM_LANDMARKS, ROI_DATASET_PATH
)

def get_roi_from_landmarks(landmarks, original_shape, padding_factor=0.1):
    """Calculates a bounding box from landmarks and adds padding."""
    # Use a subset of landmarks that frame the neck area
    # Indices for Sella(10), Gonion(14), Menton(3)
    # These indices are 0-based from the list of 29 landmarks
    relevant_indices = [10, 14, 3] 
    points = landmarks[relevant_indices]
    
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])

    # Add padding
    width = x_max - x_min
    height = y_max - y_min
    padding_x = int(width * padding_factor)
    padding_y = int(height * padding_factor)

    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(original_shape[1], x_max + padding_x)
    y_max = min(original_shape[0], y_max + padding_y)

    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # --- 1. Load Landmark Detection Model ---
    print("Loading landmark detection model...")
    landmark_model_path = 'resnet18_256x256_mre18.4.pth'
    if not os.path.exists(landmark_model_path):
        print(f"Error: Landmark model not found at {landmark_model_path}")
        return

    landmark_model = AdvancedCephNet().to(DEVICE)
    # The saved model is a state_dict, not a full model object
    landmark_model.load_state_dict(torch.load(landmark_model_path, map_location=DEVICE))
    landmark_model.eval()
    print("Landmark model loaded successfully.")

    # --- 2. Define Paths ---
    output_roi_dataset_path = ROI_DATASET_PATH
    print(f"Output ROI dataset will be saved to: {output_roi_dataset_path}")

    # --- 3. Process Each Dataset Split (Train, Valid, Test) ---
    for split in ["train", "valid", "test"]:
        print(f"\nProcessing '{split}' set...")

        # Input dataset
        dataset = AarizDataset(dataset_folder_path=DATASET_PATH, mode=split.upper(), image_size=IMAGE_SIZE)
        if not len(dataset):
            print(f"Split '{split}' is empty, skipping.")
            continue

        # Create output directories
        output_image_dir = os.path.join(output_roi_dataset_path, split, "Cephalograms")
        output_annot_dir = os.path.join(output_roi_dataset_path, split, "Annotations", "CVM Stages")
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_annot_dir, exist_ok=True)

        for i in tqdm(range(len(dataset)), desc=f"Creating ROIs for {split}"):
            # Get data for landmark prediction
            image_tensor, _, _ = dataset[i]
            image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

            # Predict landmarks
            with torch.no_grad():
                landmarks_pred_flat, _ = landmark_model(image_tensor)
            
            landmarks_pred = landmarks_pred_flat.view(NUM_LANDMARKS, 2).cpu().numpy()

            # Load original image to get full resolution
            original_image_path = os.path.join(dataset.images_root_path, dataset.images_list[i])
            original_image = cv2.imread(original_image_path)
            if original_image is None:
                print(f"Warning: Could not read {original_image_path}, skipping.")
                continue
            
            # Scale predicted landmarks to original image size
            scale_x = original_image.shape[1] / IMAGE_SIZE[1]
            scale_y = original_image.shape[0] / IMAGE_SIZE[0]
            landmarks_scaled = landmarks_pred * np.array([scale_x, scale_y])

            # Get ROI bounding box
            x, y, w, h = get_roi_from_landmarks(landmarks_scaled, original_image.shape)

            # Crop original image to ROI
            roi_image = original_image[y:y+h, x:x+w]

            # Save cropped ROI image
            output_image_path = os.path.join(output_image_dir, dataset.images_list[i])
            cv2.imwrite(output_image_path, roi_image)

            # Copy corresponding CVM annotation file
            label_file_name = dataset.images_list[i].split(".")[0] + ".json"
            source_annot_path = os.path.join(dataset.cvm_annotations_root, label_file_name)
            dest_annot_path = os.path.join(output_annot_dir, label_file_name)
            if os.path.exists(source_annot_path):
                with open(source_annot_path, 'r') as f_in, open(dest_annot_path, 'w') as f_out:
                    f_out.write(f_in.read())

    print("\nROI dataset creation complete.")

if __name__ == '__main__':
    main()
