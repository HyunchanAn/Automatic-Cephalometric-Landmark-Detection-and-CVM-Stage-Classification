import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import albumentations as A

# Important: We must use the same model and dataset class used for training the best model.
# The original HeatmapModel (not UNet) was the best.
from model import UNetHeatmapModel # Use the latest model definition
from dataset import HeatmapDataset
from config import DATASET_PATH, NUM_LANDMARKS, NUM_CVM_STAGES

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Use settings from the best model (MRE 4.5px) ---
MODEL_PATH = 'models/model_heatmap_resnet50_finetuned_mre4.5.pth'
IMAGE_SIZE_PRED = (256, 256)
HEATMAP_OUTPUT_SIZE_PRED = (64, 64) # 256 / 4 = 64
BATCH_SIZE_PRED = 16

OUTPUT_CSV_PATH = 'landmark_predictions.csv'

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

def get_true_cvm_stage(image_filename, split):
    """Helper to read the true CVM stage from the annotation file."""
    json_filename = image_filename.split('.')[0] + '.json'
    file_path = os.path.join(DATASET_PATH, split, "Annotations", "CVM Stages", json_filename)
    try:
        with open(file_path, mode="r") as f:
            cvm_annotations = json.load(f)
        return cvm_annotations["cvm_stage"]["value"]
    except FileNotFoundError:
        return None

def main():
    # --- 1. Load Model ---
    print(f"Loading model from {MODEL_PATH}...")
    model = UNetHeatmapModel(num_landmarks=NUM_LANDMARKS).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return

    # Load the state dict from the old model
    pretrained_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model_dict = model.state_dict()

    # 1. Filter out unnecessary keys, and keys with shape mismatches
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    # 2. Overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. Load the new state dict
    model.load_state_dict(model_dict)
    model.eval()
    print(f"Model loaded successfully. Transferred {len(pretrained_dict)} matching layers.")

    # --- 2. Prepare Data and Predict ---
    results = []
    splits = ['train', 'valid', 'test']

    for split in splits:
        print(f"\nProcessing split: {split}...")
        dataset = HeatmapDataset(
            dataset_folder_path=DATASET_PATH, 
            mode=split.upper(), 
            output_size=HEATMAP_OUTPUT_SIZE_PRED,
            # Use the settings that match the loaded model
        )
        # We need to manually override the dataset's image_size for prediction
        dataset.image_size = IMAGE_SIZE_PRED
        # Also adjust the transform to use the correct size
        dataset.transform.transforms[0] = A.Resize(height=IMAGE_SIZE_PRED[0], width=IMAGE_SIZE_PRED[1])

        loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE_PRED, shuffle=False)

        for i, (images, heatmaps_true, landmarks_true) in enumerate(tqdm(loader, desc=f"Predicting {split}")):
            images = images.to(DEVICE)
            
            with torch.no_grad():
                heatmaps_pred = model(images)
            
            coords_pred = get_coords_from_heatmaps(heatmaps_pred, IMAGE_SIZE_PRED, HEATMAP_OUTPUT_SIZE_PRED).cpu().numpy()
            landmarks_true = landmarks_true.cpu().numpy()

            # Get image filenames for this batch
            start_index = i * BATCH_SIZE_PRED
            end_index = start_index + len(images)
            batch_image_files = dataset.images_list[start_index:end_index]

            for j in range(len(batch_image_files)):
                image_name = batch_image_files[j]
                true_cvm = get_true_cvm_stage(image_name, split)
                for k in range(NUM_LANDMARKS):
                    results.append({
                        'image_name': image_name,
                        'split': split,
                        'landmark_index': k + 1,
                        'pred_x': coords_pred[j, k, 0],
                        'pred_y': coords_pred[j, k, 1],
                        'true_x': landmarks_true[j, k, 0],
                        'true_y': landmarks_true[j, k, 1],
                        'cvm_stage': true_cvm
                    })

    # --- 3. Save to CSV ---
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nSuccessfully saved {len(df)} predictions to {OUTPUT_CSV_PATH}")

if __name__ == '__main__':
    main()
