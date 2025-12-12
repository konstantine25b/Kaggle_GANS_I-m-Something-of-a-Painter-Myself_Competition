import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import glob

class ImageDataset(Dataset):
    def __init__(self, root_monet, root_photo, transform=None):
        self.transform = transform
        self.monet_files = sorted(glob.glob(os.path.join(root_monet, "*.*")))
        self.photo_files = sorted(glob.glob(os.path.join(root_photo, "*.*")))
        
        # Make sure we can iterate even if lengths differ
        self.len_monet = len(self.monet_files)
        self.len_photo = len(self.photo_files)
        self.length = max(self.len_monet, self.len_photo)

    def __getitem__(self, index):
        # Use modulo to loop over smaller dataset
        monet_idx = index % self.len_monet
        photo_idx = index % self.len_photo

        monet_path = self.monet_files[monet_idx]
        photo_path = self.photo_files[photo_idx]

        monet_img = Image.open(monet_path).convert('RGB')
        photo_img = Image.open(photo_path).convert('RGB')

        if self.transform:
            monet_img = self.transform(monet_img)
            photo_img = self.transform(photo_img)

        return {'monet': monet_img, 'photo': photo_img}

    def __len__(self):
        return self.length

import torchvision.transforms as transforms

def get_transforms(load_size=286, crop_size=256):
    transform_list = [
        transforms.Resize((load_size, load_size), Image.BICUBIC),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    return transforms.Compose(transform_list)

