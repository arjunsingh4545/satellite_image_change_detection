import torch
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import numpy as np


def model_predict(model, test_loader, device, sample_count=5):
    model.eval()
    predictions = []
    imageA_list = []
    imageB_list = []
    mask_list = []

    with torch.no_grad():
        for (imageB, imageA), mask in tqdm(test_loader, desc="Generating Predictions"):
            imageB = imageB.to(device)
            imageA = imageA.to(device)
            mask = mask.to(device)

            output = model(imageB, imageA)
            output = torch.sigmoid(output)

            predictions.extend(output.cpu().numpy())  # shape: [B, 1, H, W]
            imageA_list.extend(imageA.cpu().numpy())  # shape: [B, 3, H, W]
            imageB_list.extend(imageB.cpu().numpy())  # shape: [B, 3, H, W]
            mask_list.extend(mask.cpu().numpy())  # shape: [B, H, W]

    # Random sampling from the accumulated lists
    total_samples = len(predictions)
    sample_indices = random.sample(
        range(total_samples), min(sample_count, total_samples)
    )

    sampled_predictions = [predictions[i] for i in sample_indices]
    sampled_imageA = [imageA_list[i] for i in sample_indices]
    sampled_imageB = [imageB_list[i] for i in sample_indices]
    sampled_mask = [mask_list[i] for i in sample_indices]

    return sampled_predictions, sampled_imageA, sampled_imageB, sampled_mask


def plot_predictions(predictions, imageA_list, imageB_list, mask_list, num_images=5):
    for i in range(min(num_images, len(predictions))):
        pred = predictions[i][0]  # [1, H, W] → [H, W]
        imgA = imageA_list[i].transpose(1, 2, 0)  # [3, H, W] → [H, W, 3]
        imgB = imageB_list[i].transpose(1, 2, 0)
        gt_mask = mask_list[i]  # [H, W]

        # Normalize images to [0, 1] for proper visualization
        imgA = (imgA - imgA.min()) / (imgA.max() - imgA.min() + 1e-6)
        imgB = (imgB - imgB.min()) / (imgB.max() - imgB.min() + 1e-6)

        plt.figure(figsize=(16, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(imgA)
        plt.title(f"Image A - Sample {i+1}")
        plt.axis("off")
        plt.xlabel("Image A")

        plt.subplot(1, 4, 2)
        plt.imshow(imgB)
        plt.title("Image B")
        plt.axis("off")
        plt.xlabel("Image B")

        plt.subplot(1, 4, 3)
        plt.imshow(gt_mask, cmap="gray")
        plt.title("Ground Truth Mask")
        plt.axis("off")
        plt.xlabel("Ground Truth Mask")

        plt.subplot(1, 4, 4)
        plt.imshow(pred > 0.5, cmap="gray")  # Thresholded prediction
        plt.title("Predicted Mask")
        plt.axis("off")
        plt.xlabel("Predicted Mask")

        # plt.tight_layout()
        plt.show()


import albumentations as A
from albumentations.pytorch import ToTensorV2


def test_and_plot_single_sample(
    model,
    imgA_path,
    imgB_path,
    label_path,
    device,
    image_size=(256, 256),
    threshold=0.5,
):
    def albumentations_transform(imageA, imageB, label):
        transform = A.Compose(
            [
                A.Resize(256, 256),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
                ToTensorV2(),
            ],
            additional_targets={"image1": "image", "mask": "mask"},
        )

        augmented = transform(image=imageB, image1=imageA, mask=label)
        return augmented["image"], augmented["image1"], augmented["mask"]

    # Load images
    imgA_pil = Image.open(imgA_path).convert("RGB")
    imgB_pil = Image.open(imgB_path).convert("RGB")
    label_pil = Image.open(label_path).convert("L")

    imgA_np = np.array(imgA_pil)
    imgB_np = np.array(imgB_pil)
    label_np = np.array(label_pil)

    imgA_tensor, imgB_tensor, label_tensor = albumentations_transform(
        imgA_np, imgB_np, label_np
    )
    imgA_tensor = imgA_tensor.unsqueeze(0).to(device)
    imgB_tensor = imgB_tensor.unsqueeze(0).to(device)
    label_tensor = label_tensor.unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(imgB_tensor, imgA_tensor)
        output = torch.sigmoid(output)
        prediction = (output > threshold).float()

    # Metrics
    TP = torch.sum(prediction * label_tensor)
    FP = torch.sum(prediction * (1 - label_tensor))
    FN = torch.sum((1 - prediction) * label_tensor)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    # Convert prediction to numpy
    pred_mask = prediction.squeeze().cpu().numpy()

    # Plot
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(imgA_np)
    axs[0].set_title("Image A (Before)")
    axs[1].imshow(imgB_np)
    axs[1].set_title("Image B (After)")
    axs[2].imshow(label_np, cmap="gray")
    axs[2].set_title("Ground Truth Mask")
    axs[3].imshow(pred_mask, cmap="gray")
    axs[3].set_title("Predicted Mask")

    for ax in axs:
        ax.axis("off")

    plt.suptitle(
        f"Precision: {precision.item():.4f}, Recall: {recall.item():.4f}", fontsize=14
    )
    plt.tight_layout()
    plt.show()

    return pred_mask, precision.item(), recall.item()
