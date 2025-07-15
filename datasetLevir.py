import os
import torch
from PIL import Image
import numpy as np
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import psutil
dataset_base_path = r'/location/to/levir-cd/dataset'
train_images_path = os.path.join(dataset_base_path , 'train')
test_images_path = os.path.join(dataset_base_path, 'test')
val_images_path = os.path.join(dataset_base_path, 'val')

# image size is set to 256 due to GPU constraints , you can modify it as per your use
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256

class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir , transform=None):
        self.images_dir_a = os.path.join(images_dir, 'A')
        self.images_dir_b = os.path.join(images_dir, 'B')
        self.mask_dir = os.path.join(images_dir, 'label')
        self.transform = transform
        self.images_a_list = sorted(f for f in os.listdir(self.images_dir_a) if f.endswith(('.png', '.jpg', '.jpeg')))
        self.images_b_list = sorted(f for f in os.listdir(self.images_dir_b) if f.endswith(('.png', '.jpg', '.jpeg')))
        self.masks_list = sorted(f for f in os.listdir(self.mask_dir) if f.endswith(('.png', '.jpg', '.jpeg')))
        assert len(self.images_a_list) == len(self.images_b_list) == len(self.masks_list), \
            "Mismatch in number of images and masks"
    def __len__(self):
        return len(self.images_b_list)

    def __getitem__(self, idx):
        img_path_b = os.path.join(self.images_dir_b, self.images_b_list[idx])
        img_path_a = os.path.join(self.images_dir_a, self.images_a_list[idx])
        mask_path = os.path.join(self.mask_dir, self.masks_list[idx])

        image_b = Image.open(img_path_b).convert('RGB')
        image_a = Image.open(img_path_a).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        #for albumentation, we need to convert PIL images to numpy arrays
        image_b = np.array(image_b)
        image_a = np.array(image_a)
        mask = np.array(mask)

        if self.transform:
            augmented = self.transform(image=image_b, image1=image_a, mask=mask)
            image_b = augmented['image']
            image_a = augmented['image1']
            mask = augmented['mask']
            mask = (mask > 0.5).float()
        else:
            image_b = torch.tensor(image_b / 255.0, dtype=torch.float32).permute(2, 0, 1)
            image_a = torch.tensor(image_a / 255.0, dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor((mask > 0).astype(np.float32)).unsqueeze(0)  # [1, H, W]
        return (image_b, image_a), mask


# albumentation transforms (data augmentation)
train_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5), # Added Vertical Flip
    A.Rotate(limit=45, p=0.5, interpolation=cv2.INTER_NEAREST), # Added rotation
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), # Added brightness/contrast
    A.GaussNoise(p=0.2), # Added Gaussian Noise
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),  # Normalize to [0, 1]
    A.pytorch.ToTensorV2()
], additional_targets={'image1': 'image'}) # 'image1' corresponds to image_a

val_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),  # Normalize to [0, 1]
    A.pytorch.ToTensorV2()
], additional_targets={'image1': 'image'}) # 'image1' corresponds to image_a


#creating datasets for dataloader
train_dataset = CreateDataset(train_images_path, transform=train_transform)
val_dataset = CreateDataset(val_images_path, transform=val_transform)
test_dataset = CreateDataset(test_images_path, transform=val_transform)

# The datasets can now be used with DataLoader for training, validation, and testing
def myDataLoader(dataset, shuffle=True, BATCH_SIZE=16, NUM_WORKERS=2):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True 
    )

train_loader = myDataLoader(train_dataset, shuffle=True, BATCH_SIZE=4, NUM_WORKERS=2)
val_loader = myDataLoader(val_dataset, shuffle=False, BATCH_SIZE=4, NUM_WORKERS=2)
test_loader = myDataLoader(test_dataset, shuffle=False, BATCH_SIZE=4, NUM_WORKERS=2)


def visualize_sample(dataset, idx=0):
    (img_b, img_a), mask = dataset[idx]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_a.permute(1, 2, 0).numpy())
    axs[0].set_title("Image A")
    axs[1].imshow(img_b.permute(1, 2, 0).numpy())
    axs[1].set_title("Image B")
    axs[2].imshow(mask.squeeze().numpy(), cmap='gray')
    axs[2].set_title("Mask")
    plt.show()



