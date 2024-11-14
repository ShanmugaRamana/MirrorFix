# scripts/dataset_loader.py
import os
from torch.utils.data import Dataset
from PIL import Image
from utils.transformations import ReflectionTransforms

class ReflectionDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        
        self.input_images = [f for f in os.listdir(input_dir) 
                           if f.lower().endswith(('.png', '.jpg'))]
        
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        target_path = os.path.join(self.target_dir, self.input_images[idx])
        
        input_image = Image.open(input_path).convert('RGB')
        target_image = Image.open(target_path).convert('RGB')
        
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        
        return input_image, target_image