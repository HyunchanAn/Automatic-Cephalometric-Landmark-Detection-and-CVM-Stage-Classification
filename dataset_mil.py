import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import random
import json
import cv2
import numpy as np

class MILDataset(Dataset):
    def __init__(self, mode, dataset_path="Aariz", transform=None, num_patches=16, patch_size=128):
        if mode.upper() not in ["TRAIN", "VALID", "TEST"]:
            raise ValueError("mode could only be TRAIN, VALID or TEST")
        
        self.mode = mode.lower()
        self.dataset_path = dataset_path
        self.transform = transform
        self.num_patches = num_patches
        self.patch_size = patch_size

        self.images_root = os.path.join(self.dataset_path, self.mode, "Cephalograms")
        self.labels_root = os.path.join(self.dataset_path, self.mode, "Annotations", "CVM Stages")
        
        self.image_files = sorted(os.listdir(self.images_root))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Get image and label paths
        image_filename = self.image_files[idx]
        label_filename = os.path.splitext(image_filename)[0] + ".json"
        
        image_path = os.path.join(self.images_root, image_filename)
        label_path = os.path.join(self.labels_root, label_filename)

        # 2. Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_pil_image = Image.fromarray(image)
        
        # 3. Load CVM Stage
        with open(label_path, 'r') as f:
            label_data = json.load(f)
        # Subtract 1 to make it 0-indexed (0-5 for stages 1-6)
        cvm_stage = label_data['cvm_stage']['value'] - 1

        w, h = original_pil_image.size

        # 4. Define patch sampling area (lower 50% of the image)
        sampling_y_min = h // 2
        
        patches = []
        for _ in range(self.num_patches):
            # 5. Sample a patch
            rand_x = random.randint(0, w - self.patch_size)
            rand_y = random.randint(sampling_y_min, h - self.patch_size)
            
            patch = original_pil_image.crop((rand_x, rand_y, rand_x + self.patch_size, rand_y + self.patch_size))
            
            if self.transform:
                patch = self.transform(patch)
            
            patches.append(patch)
            
        patches_tensor = torch.stack(patches)
        
        return patches_tensor, torch.tensor(cvm_stage, dtype=torch.long)

if __name__ == '__main__':
    # Test code
    from torchvision import transforms
    import matplotlib.pyplot as plt

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    mil_train_dataset = MILDataset(mode='TRAIN', transform=transform)

    print(f"Total training samples: {len(mil_train_dataset)}")

    patches_tensor, cvm_stage = mil_train_dataset[0]
    print(f"Shape of patches tensor: {patches_tensor.shape}")
    print(f"CVM Stage (0-indexed): {cvm_stage.item()}")

    first_patch = patches_tensor[0].permute(1, 2, 0).numpy()
    first_patch = (first_patch * 0.5) + 0.5 # un-normalize
    plt.imshow(first_patch)
    plt.title(f"Sample Patch 1 (CVM Stage: {cvm_stage.item() + 1})")
    plt.show()