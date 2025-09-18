from config import *
import numpy as np
import json
import cv2
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class AarizDataset(Dataset):
    
    def __init__(self, dataset_folder_path: str, mode: str, transform=None):
        
        if (mode == "TRAIN") or (mode == "VALID") or (mode == "TEST"):
            mode = mode.lower()
        else:
            raise ValueError("mode could only be TRAIN, VALID or TEST")
        
        self.transform = transform
        
        self.images_root_path = os.path.join(dataset_folder_path, mode, "Cephalograms")
        self.labels_root_path = os.path.join(dataset_folder_path, mode, "Annotations")
        
        self.senior_annotations_root = os.path.join(self.labels_root_path, "Cephalometric Landmarks", "Senior Orthodontists")
        self.junior_annotations_root = os.path.join(self.labels_root_path, "Cephalometric Landmarks", "Junior Orthodontists")
        self.cvm_annotations_root = os.path.join(self.labels_root_path, "CVM Stages")
        
        self.images_list = os.listdir(self.images_root_path)
        
    
    def __getitem__(self, index):
        image_file_name = self.images_list[index]
        label_file_name = self.images_list[index].split(".")[0] + "." + "json"
        
        image = self.get_image(image_file_name)
        original_height, original_width, _ = image.shape # Get original dimensions

        landmarks = self.get_landmarks(label_file_name)
        cvm_stage = self.get_cvm_stage(label_file_name)

        if self.transform:
            image = self.transform(image)
            # Scale landmarks to the new image size (IMAGE_SIZE from config)
            # Assuming self.transform includes transforms.Resize(IMAGE_SIZE)
            target_height, target_width = IMAGE_SIZE # IMAGE_SIZE is (height, width)
            
            scale_x = target_width / original_width
            scale_y = target_height / original_height
            
            landmarks[:, 0] = landmarks[:, 0] * scale_x
            landmarks[:, 1] = landmarks[:, 1] * scale_y
        
        # Flatten landmarks for regression output
        landmarks = landmarks.flatten()

        return image, torch.from_numpy(landmarks).float(), torch.from_numpy(cvm_stage).float()
    
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
        landmarks[:, 0] = np.ceil((0.5) * (junior_annotations[:, 0] + senior_annotations[:, 0]))
        landmarks[:, 1] = np.ceil((0.5) * (junior_annotations[:, 1] + senior_annotations[:, 1]))
        
        return np.array(landmarks, dtype=np.float32)
    
    def get_cvm_stage(self, file_name):
        file_path = os.path.join(self.cvm_annotations_root, file_name)
        
        with open(file_path, mode="r") as file:
            cvm_annotations = json.load(file)
        
        cvm_stage_value = cvm_annotations["cvm_stage"]["value"]
        cvm_stage = np.zeros(shape=(NUM_CVM_STAGES, ))
        cvm_stage[cvm_stage_value - 1] = 1.0
        
        return np.array(cvm_stage, dtype=np.float32)
    
    def __len__(self):
        return len(self.images_list)