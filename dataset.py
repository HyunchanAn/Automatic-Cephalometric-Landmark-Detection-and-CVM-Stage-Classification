from config import *
import numpy as np
import json
import cv2
import os
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AarizDataset(Dataset):
    
    def __init__(self, dataset_folder_path: str, mode: str, image_size: tuple):
        
        if (mode == "TRAIN") or (mode == "VALID") or (mode == "TEST"):
            self.mode = mode.lower()
        else:
            raise ValueError("mode could only be TRAIN, VALID or TEST")
        
        # Define Albumentations pipelines
        if self.mode == 'train':
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                # --- DEBUGGING: Re-enabling Affine to isolate error ---
                A.Affine(
                    scale=(0.95, 1.05), # Range reduced for stability
                    translate_percent=(-0.03, 0.03), # Range reduced for stability
                    rotate=(-10, 10), # Range reduced for stability
                    p=0.7
                ),
                # A.ElasticTransform(p=0.5), # DISABLED: Caused dataloader errors.
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                A.GaussNoise(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', label_fields=[], remove_invisible=False))
        else: # For 'valid' and 'test'
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', label_fields=[], remove_invisible=False))

        self.images_root_path = os.path.join(dataset_folder_path, self.mode, "Cephalograms")
        self.labels_root_path = os.path.join(dataset_folder_path, self.mode, "Annotations")
        
        self.senior_annotations_root = os.path.join(self.labels_root_path, "Cephalometric Landmarks", "Senior Orthodontists")
        self.junior_annotations_root = os.path.join(self.labels_root_path, "Cephalometric Landmarks", "Junior Orthodontists")
        self.cvm_annotations_root = os.path.join(self.labels_root_path, "CVM Stages")
        
        self.images_list = os.listdir(self.images_root_path)
        
    
    def __getitem__(self, index):
        image_file_name = self.images_list[index]
        label_file_name = self.images_list[index].split(".")[0] + "." + "json"
        
        image = self.get_image(image_file_name)
        landmarks = self.get_landmarks(label_file_name)
        cvm_stage = self.get_cvm_stage(label_file_name)

        # Apply albumentations transform
        transformed = self.transform(image=image, keypoints=landmarks)
        image = transformed['image']
        landmarks = transformed['keypoints']
        
        # Flatten landmarks for regression output
        landmarks = np.array(landmarks, dtype=np.float32).flatten()

        return image, torch.from_numpy(landmarks).float(), torch.tensor(cvm_stage, dtype=torch.long)
    
    def get_image(self, file_name: str):
        file_path = os.path.join(self.images_root_path, file_name)
        
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return np.array(image, dtype=np.uint8)
    
    
    def get_landmarks(self, file_name):
        file_path = os.path.join(self.senior_annotations_root, file_name)
        with open(file_path, mode="r") as file:
            senior_annotations = json.load(file)
        
        senior_annotations = [[landmark["value"]["x"], landmark["value"]["y"]] for landmark in senior_annotations["landmarks"]]
        senior_annotations = np.array(senior_annotations, dtype=np.float32)
        
        file_path = os.path.join(self.junior_annotations_root, file_name)
        with open(file_path, mode="r") as file:
            junior_annotations = json.load(file)

        junior_annotations = [[landmark["value"]["x"], landmark["value"]["y"]] for landmark in junior_annotations["landmarks"]]
        junior_annotations = np.array(junior_annotations, dtype=np.float32)
        
        landmarks = np.zeros(shape=(NUM_LANDMARKS, 2), dtype=np.float64)
        landmarks[:, 0] = (0.5) * (junior_annotations[:, 0] + senior_annotations[:, 0])
        landmarks[:, 1] = (0.5) * (junior_annotations[:, 1] + senior_annotations[:, 1])
        
        return np.array(landmarks, dtype=np.float32)
    
    def get_cvm_stage(self, file_name):
        file_path = os.path.join(self.cvm_annotations_root, file_name)
        
        with open(file_path, mode="r") as file:
            cvm_annotations = json.load(file)
        
        cvm_stage_value = cvm_annotations["cvm_stage"]["value"]
        # Return integer index (0-5) instead of one-hot vector
        return cvm_stage_value - 1
    
    def __len__(self):
        return len(self.images_list)