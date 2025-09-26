import numpy as np
import json
import cv2
import os
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import NUM_LANDMARKS # IMAGE_SIZE is now passed as an argument

class HeatmapDataset(Dataset):
    
    def __init__(self, dataset_folder_path: str, mode: str, image_size: tuple, output_size: tuple = (64, 64), sigma: int = 2):
        
        if mode.upper() not in ["TRAIN", "VALID", "TEST"]:
            raise ValueError("mode could only be TRAIN, VALID or TEST")
        self.mode = mode.lower()

        self.image_size = image_size # Use passed image_size
        self.output_size = output_size
        self.sigma = sigma

        # Define Albumentations pipelines
        if self.mode == 'train':
            self.transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                A.GaussNoise(p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.7),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else: # For 'valid' and 'test'
            self.transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        self.images_root_path = os.path.join(dataset_folder_path, self.mode, "Cephalograms")
        self.labels_root_path = os.path.join(dataset_folder_path, self.mode, "Annotations")
        self.senior_annotations_root = os.path.join(self.labels_root_path, "Cephalometric Landmarks", "Senior Orthodontists")
        self.junior_annotations_root = os.path.join(self.labels_root_path, "Cephalometric Landmarks", "Junior Orthodontists")
        
        self.images_list = os.listdir(self.images_root_path)
        
    def __getitem__(self, index):
        image_file_name = self.images_list[index]
        label_file_name = image_file_name.split(".")[0] + ".json"
        
        image = self._get_image(image_file_name)
        landmarks = self._get_landmarks(label_file_name)

        # Apply albumentations transform
        transformed = self.transform(image=image, keypoints=landmarks)
        image = transformed['image']
        landmarks = transformed['keypoints']

        # Generate heatmaps
        heatmaps = self._generate_heatmaps(landmarks)
        
        return image, torch.tensor(heatmaps, dtype=torch.float32), torch.tensor(landmarks, dtype=torch.float32)

    def _generate_heatmaps(self, landmarks):
        heatmaps = np.zeros((NUM_LANDMARKS, self.output_size[0], self.output_size[1]), dtype=np.float32)
        scale_x = self.output_size[1] / self.image_size[1]
        scale_y = self.output_size[0] / self.image_size[0]

        for i, (x, y) in enumerate(landmarks):
            # Scale landmark coordinates to heatmap size
            hm_x = int(x * scale_x)
            hm_y = int(y * scale_y)

            if 0 <= hm_x < self.output_size[1] and 0 <= hm_y < self.output_size[0]:
                heatmaps[i] = self._create_gaussian_heatmap(hm_x, hm_y)
        
        return heatmaps

    def _create_gaussian_heatmap(self, center_x, center_y):
        heatmap = np.zeros((self.output_size[0], self.output_size[1]), dtype=np.float32)
        tmp_size = self.sigma * 3
        
        # Generate gaussian region
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

        # Usable gaussian range
        left = min(center_x, tmp_size)
        right = min(self.output_size[1] - center_x, tmp_size + 1)
        top = min(center_y, tmp_size)
        bottom = min(self.output_size[0] - center_y, tmp_size + 1)

        # Cropped gaussian
        cropped_g = g[y0 - top:y0 + bottom, x0 - left:x0 + right]
        # Target heatmap region
        paste_y1, paste_y2 = center_y - top, center_y + bottom
        paste_x1, paste_x2 = center_x - left, center_x + right

        heatmap[paste_y1:paste_y2, paste_x1:paste_x2] = cropped_g
        return heatmap

    def _get_image(self, file_name: str):
        file_path = os.path.join(self.images_root_path, file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.array(image, dtype=np.uint8)
    
    def _get_landmarks(self, file_name):
        file_path = os.path.join(self.senior_annotations_root, file_name)
        with open(file_path, mode="r") as file:
            senior_annotations = json.load(file)
        
        senior_annotations = [[landmark["value"]["x"], landmark["value"]["y"]] for landmark in senior_annotations["landmarks"]]
        
        file_path = os.path.join(self.junior_annotations_root, file_name)
        with open(file_path, mode="r") as file:
            junior_annotations = json.load(file)

        junior_annotations = [[landmark["value"]["x"], landmark["value"]["y"]] for landmark in junior_annotations["landmarks"]]
        
        landmarks = np.zeros(shape=(NUM_LANDMARKS, 2), dtype=np.float32)
        for i in range(NUM_LANDMARKS):
            landmarks[i, 0] = np.ceil((0.5) * (junior_annotations[i][0] + senior_annotations[i][0]))
            landmarks[i, 1] = np.ceil((0.5) * (junior_annotations[i][1] + senior_annotations[i][1]))
        
        return landmarks
    
    def __len__(self):
        return len(self.images_list)