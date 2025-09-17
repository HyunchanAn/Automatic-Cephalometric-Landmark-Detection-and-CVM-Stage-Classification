import torch
import cv2
import numpy as np
import argparse
import os
from torchvision import transforms

from model import AdvancedCephNet
from dataset import AarizDataset
from config import CHECKPOINT_PATH, NUM_LANDMARKS, CVM_STAGES, DATASET_PATH, IMAGE_SIZE

def main(args):
    # --- 1. Setup ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # --- 2. Load Model ---
    print("Loading model...")
    model = AdvancedCephNet().to(DEVICE)

    if not os.path.exists(CHECKPOINT_PATH):
        print("Error: Checkpoints directory not found. Please train the model first.")
        return

    checkpoint_list = os.listdir(CHECKPOINT_PATH)
    if not checkpoint_list:
        print("Error: No checkpoints found. Please train the model first.")
        return

    latest_checkpoint = max(checkpoint_list, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, latest_checkpoint), map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: {latest_checkpoint}")

    # --- 3. Load Data ---
    # We need the original image for visualization, and the transformed image for the model
    image_index = args.index

    # Transformations for the model input (must be same as validation)
    model_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset object to get image path and labels
    # No transform here, as we load the original image ourselves
    valid_dataset = AarizDataset(dataset_folder_path=DATASET_PATH, mode="VALID")
    
    if image_index >= len(valid_dataset):
        print(f"Error: Index {image_index} is out of bounds for the validation set (size: {len(valid_dataset)}).")
        return

    # Get the original image path and load it
    image_name = valid_dataset.images_list[image_index]
    image_path = os.path.join(valid_dataset.images_root_path, image_name)
    original_image = cv2.imread(image_path)
    
    # Get the labels and prepare the image for the model
    image_for_model, landmarks_true_flat, cvm_stage_true_onehot = valid_dataset[image_index]
    image_tensor = model_transform(image_for_model).unsqueeze(0).to(DEVICE)

    # --- 4. Run Inference ---
    print("Running inference...")
    with torch.no_grad():
        landmarks_pred_flat, cvm_stage_pred_logits = model(image_tensor)

    # Process predictions
    landmarks_pred = landmarks_pred_flat.view(NUM_LANDMARKS, 2).cpu().numpy()
    landmarks_true = landmarks_true_flat.view(NUM_LANDMARKS, 2).cpu().numpy()
    
    cvm_pred_idx = cvm_stage_pred_logits.argmax(dim=1).item()
    cvm_true_idx = cvm_stage_true_onehot.argmax().item()

    # Find CVM stage title from index
    cvm_pred_title = next((stage["title"] for stage in CVM_STAGES.values() if stage["value"] == cvm_pred_idx + 1), "Unknown")
    cvm_true_title = next((stage["title"] for stage in CVM_STAGES.values() if stage["value"] == cvm_true_idx + 1), "Unknown")

    # --- 5. Visualize ---
    print("Creating visualization...")
    vis_image = original_image.copy()
    
    # Draw landmarks
    for i in range(NUM_LANDMARKS):
        # True landmarks in Green
        true_x, true_y = int(landmarks_true[i, 0]), int(landmarks_true[i, 1])
        cv2.circle(vis_image, (true_x, true_y), radius=5, color=(0, 255, 0), thickness=-1)

        # Predicted landmarks in Red
        pred_x, pred_y = int(landmarks_pred[i, 0]), int(landmarks_pred[i, 1])
        cv2.circle(vis_image, (pred_x, pred_y), radius=5, color=(0, 0, 255), thickness=-1)
        # Draw line between true and predicted
        cv2.line(vis_image, (true_x, true_y), (pred_x, pred_y), color=(255, 255, 0), thickness=1)


    # Add text for CVM stages
    cv2.putText(vis_image, f"True CVM: {cvm_true_title}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(vis_image, f"Pred CVM: {cvm_pred_title}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Add legend
    cv2.circle(vis_image, (10, 110), 5, (0, 255, 0), -1)
    cv2.putText(vis_image, "True Landmark", (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.circle(vis_image, (10, 140), 5, (0, 0, 255), -1)
    cv2.putText(vis_image, "Predicted Landmark", (25, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Save the output image
    output_filename = "visualization_output.png"
    cv2.imwrite(output_filename, vis_image)
    print(f"Visualization saved to {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize model predictions on a single image.")
    parser.add_argument('--index', type=int, default=0, help='Index of the image in the validation set to visualize.')
    args = parser.parse_args()
    main(args)
