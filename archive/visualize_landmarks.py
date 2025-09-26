import cv2
import numpy as np
import json
import os

from config import DATASET_PATH, NUM_LANDMARKS

# --- Configuration ---
# We'll pick the first image from the training set for visualization
IMAGE_TO_VISUALIZE = os.listdir(os.path.join(DATASET_PATH, 'train', 'Cephalograms'))[0]
OUTPUT_IMAGE_PATH = 'landmark_visualization.jpg'

# --- Helper functions (adapted from dataset.py) ---
def get_image(file_name: str):
    file_path = os.path.join(DATASET_PATH, 'train', 'Cephalograms', file_name)
    image = cv2.imread(file_path)
    return image

def get_landmarks(file_name: str):
    senior_path = os.path.join(DATASET_PATH, 'train', 'Annotations', 'Cephalometric Landmarks', 'Senior Orthodontists', file_name)
    junior_path = os.path.join(DATASET_PATH, 'train', 'Annotations', 'Cephalometric Landmarks', 'Junior Orthodontists', file_name)
    
    with open(senior_path, mode="r") as f:
        senior_annotations = json.load(f)
    senior_annotations = [[landmark["value"]["x"], landmark["value"]["y"]] for landmark in senior_annotations["landmarks"]]
    
    with open(junior_path, mode="r") as f:
        junior_annotations = json.load(f)
    junior_annotations = [[landmark["value"]["x"], landmark["value"]["y"]] for landmark in junior_annotations["landmarks"]]
    
    landmarks = np.zeros(shape=(NUM_LANDMARKS, 2), dtype=np.float32)
    for i in range(NUM_LANDMARKS):
        landmarks[i, 0] = np.ceil((0.5) * (junior_annotations[i][0] + senior_annotations[i][0]))
        landmarks[i, 1] = np.ceil((0.5) * (junior_annotations[i][1] + senior_annotations[i][1]))
    
    return landmarks.astype(int)

def main():
    print(f"Visualizing landmarks for image: {IMAGE_TO_VISUALIZE}")
    
    # 1. Load image and landmarks
    image = get_image(IMAGE_TO_VISUALIZE)
    json_filename = IMAGE_TO_VISUALIZE.split('.')[0] + '.json'
    landmarks = get_landmarks(json_filename)
    
    if image is None:
        print("Error: Image not found.")
        return
    if landmarks is None:
        print("Error: Landmarks not found.")
        return

    # 2. Draw landmarks and their numbers on the image
    for i, (x, y) in enumerate(landmarks):
        # Draw a circle at the landmark position
        cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1) # Green dot
        
        # Put the landmark number next to the circle
        cv2.putText(
            image, 
            str(i + 1), 
            (x + 5, y + 5), # Offset text slightly
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.8, 
            color=(255, 255, 0), # Cyan text
            thickness=2
        )

    # 3. Save the output image
    cv2.imwrite(OUTPUT_IMAGE_PATH, image)
    print(f"Successfully created visualization image at: {OUTPUT_IMAGE_PATH}")

if __name__ == '__main__':
    main()
