import numpy as np
import json
import cv2
import os
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import NUM_CVM_STAGES

class RoiCvmDataset(Dataset):
    
    def __init__(self, dataset_folder_path: str, mode: str, image_size: tuple):
        
        if (mode == "TRAIN") or (mode == "VALID") or (mode == "TEST"):
            self.mode = mode.lower()
        else:
            raise ValueError("mode could only be TRAIN, VALID or TEST")
        
        # Define Albumentations pipelines
        if self.mode == 'train':
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                A.GaussNoise(p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.7),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ])
        else: # For 'valid' and 'test'
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ])

        self.images_root_path = os.path.join(dataset_folder_path, self.mode, "Cephalograms")
        self.cvm_annotations_root = os.path.join(dataset_folder_path, self.mode, "Annotations", "CVM Stages")
        
        self.images_list = os.listdir(self.images_root_path)
        
    def __getitem__(self, index):
        image_file_name = self.images_list[index]
        label_file_name = image_file_name.split(".")[0] + "." + "json"
        
        image = self.get_image(image_file_name)
        cvm_stage = self.get_cvm_stage(label_file_name)

        transformed = self.transform(image=image)
        image = transformed['image']
        
        return image, torch.tensor(cvm_stage, dtype=torch.long)
    
    def get_image(self, file_name: str):
        file_path = os.path.join(self.images_root_path, file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.array(image, dtype=np.uint8)
    
    def get_cvm_stage(self, file_name):
        file_path = os.path.join(self.cvm_annotations_root, file_name)
        
        with open(file_path, mode="r") as file:
            cvm_annotations = json.load(file)
        
        cvm_stage_value = cvm_annotations["cvm_stage"]["value"]
        return cvm_stage_value - 1 # Return 0-5 index
    
    def __len__(self):
        return len(self.images_list)
