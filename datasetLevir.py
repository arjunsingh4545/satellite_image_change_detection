import os
import torch
from PIL import Image
import numpy as np
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import psutil

# === Path Setup ===
# Base directory where the LEVIR-CD dataset is stored
dataset_base_path = r'/location/to/levir-cd/dataset'

# Sub-directories for train, test, and validation sets
train_images_path = os.path.join(dataset_base_path, 'train')
test_images_path = os.path.join(dataset_base_path, 'test')
val_images_path = os.path.join(dataset_base_path, 'val')

# Resize dimensions for images and masks (smaller size helps with limited GPU memory)
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256

# === Custom Dataset Class ===
class CreateDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for LEVIR-CD
    Loads image pairs (A, B) and corresponding binary change masks
    Applies optional Albumentations-based transforms
    """
    def __init__(self, images_dir, transform=None):
        # Directories for pre-change (A), post-change (B), and ground-truth labels
        self.images_dir_a = os.path.join(images_dir, 'A')
        self.images_dir_b = os.path.join(images_dir, 'B')
        self.mask_dir = os.path.join(images_dir, 'label')
        self.transform = transform

        # Sorted file lists for consistent pairing
        self.images_a_list = sorted(f for f in os.listdir(self.images_dir_a) if f.endswith(('.png', '.jpg', '.jpeg')))
        self.images_b_list = sorted(f for f in os.listdir(self.images_dir_b) if f.endswith(('.png', '.jpg', '.jpeg')))
        self.masks_list = sorted(f for f in os.listdir(self.mask_dir) if f.endswith(('.png', '.jpg', '.jpeg')))

        # Sanity check: Ensure 1-to-1 correspondence
        assert len(self.images_a_list) == len(self.images_b_list) == len(self.masks_list), \
            "Mismatch in number of images and masks"

    def __len__(self):
        return len(self.images_b_list)

    def __getitem__(self, idx):
        # Construct full paths for image pair and mask
        img_path_b = os.path.join(self.images_dir_b, self.images_b_list[idx])
        img_path_a = os.path.join(self.images_dir_a, self.images_a_list[idx])
        mask_path = os.path.join(self.mask_dir, self.masks_list[idx])

        # Load images and mask as PIL Images
        image_b = Image.open(img_path_b).convert('RGB')
        image_a = Image.open(img_path_a).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # grayscale mask

        # Convert to numpy arrays for Albumentations
        image_b = np.array(image_b)
        image_a = np.array(image_a)
        mask = np.array(mask)

        if self.transform:
            # Apply Albumentations transform
            augmented = self.transform(image=image_b, image1=image_a, mask=mask)
            image_b = augmented['image']
            image_a = augmented['image1']
            mask = augmented['mask']
            mask = (mask > 0.5).float()  # Binarize mask
        else:
            # Normalize and convert to torch tensors manually
            image_b = torch.tensor(image_b / 255.0, dtype=torch.float32).permute(2, 0, 1)
            image_a = torch.tensor(image_a / 255.0, dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor((mask > 0).astype(np.float32)).unsqueeze(0)  # Add channel dim

        return (image_b, image_a), mask

# === Albumentations Transforms ===

# Data augmentation for training
train_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),  # Flip vertically
    A.Rotate(limit=45, p=0.5, interpolation=cv2.INTER_NEAREST),  # Random rotation
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Lighting variation
    A.GaussNoise(p=0.2),  # Add random noise
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),  # Normalize to [0, 1]
    A.pytorch.ToTensorV2()
], additional_targets={'image1': 'image'})  # Ensure image_a is treated as an image

# For validation and testing: only resize and normalize
val_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    A.pytorch.ToTensorV2()
], additional_targets={'image1': 'image'})

# === Dataset Instantiation ===

# Create datasets for different splits
train_dataset = CreateDataset(train_images_path, transform=train_transform)
val_dataset = CreateDataset(val_images_path, transform=val_transform)
test_dataset = CreateDataset(test_images_path, transform=val_transform)

# === DataLoader Utility Function ===

def myDataLoader(dataset, shuffle=True, BATCH_SIZE=16, NUM_WORKERS=2):
    """
    Create a PyTorch DataLoader with recommended settings
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),  # Faster transfer to GPU
        persistent_workers=True  # Keep worker processes alive
    )

# Instantiate loaders
train_loader = myDataLoader(train_dataset, shuffle=True, BATCH_SIZE=4, NUM_WORKERS=2)
val_loader = myDataLoader(val_dataset, shuffle=False, BATCH_SIZE=4, NUM_WORKERS=2)
test_loader = myDataLoader(test_dataset, shuffle=False, BATCH_SIZE=4, NUM_WORKERS=2)

# === Utility for Visualization ===

def visualize_sample(dataset, idx=0):
    """
    Visualize a sample image pair (A, B) and their corresponding change mask.
    """
    (img_b, img_a), mask = dataset[idx]
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    axs[0].imshow(img_a.permute(1, 2, 0).numpy())  # Convert CHW to HWC
    axs[0].set_title("Image A")
    
    axs[1].imshow(img_b.permute(1, 2, 0).numpy())
    axs[1].set_title("Image B")
    
    axs[2].imshow(mask.squeeze().numpy(), cmap='gray')
    axs[2].set_title("Mask")
    
    plt.show()
