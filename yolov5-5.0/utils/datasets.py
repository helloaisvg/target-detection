# YOLOv5 datasets module
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import yaml

class LoadImagesAndLabels(Dataset):
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, cache_images=False):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.rect = rect
        self.cache_images = cache_images
        
        # Load dataset config
        if isinstance(path, str):
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            self.path = Path(data['path'])
            self.train_path = self.path / data['train']
        else:
            self.path = Path(path)
            self.train_path = self.path
        
        # Get image files
        self.img_files = sorted(list(self.train_path.glob('*.jpg')) + list(self.train_path.glob('*.png')))
        self.n = len(self.img_files)
        
        # Get label files (assuming YOLO format)
        self.label_files = []
        label_dir = self.path / 'labels' / 'train2017'
        if label_dir.exists():
            for img_file in self.img_files:
                label_file = label_dir / (img_file.stem + '.txt')
                self.label_files.append(label_file if label_file.exists() else None)
        else:
            self.label_files = [None] * self.n
        
        self.cache = {}
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        # Load image
        img_path = self.img_files[index]
        if img_path in self.cache:
            img = self.cache[img_path]
        else:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.cache_images:
                self.cache[img_path] = img
        
        # Resize
        h, w = img.shape[:2]
        r = self.img_size / max(h, w)
        new_w, new_h = int(w * r), int(h * r)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        top = (self.img_size - new_h) // 2
        bottom = self.img_size - new_h - top
        left = (self.img_size - new_w) // 2
        right = self.img_size - new_w - left
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Convert to tensor
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        
        # Load labels
        label_file = self.label_files[index]
        labels = []
        if label_file and label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                        # Adjust coordinates for padding
                        x = (x * new_w + left) / self.img_size
                        y = (y * new_h + top) / self.img_size
                        w = w * new_w / self.img_size
                        h = h * new_h / self.img_size
                        labels.append([int(cls), x, y, w, h])
        
        return img, torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))

