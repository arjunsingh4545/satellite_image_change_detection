import torch
import argparse
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


model_path = "/home/arjunsingh/self/projects/imageChangeDetectionFinal2/best_model.pth"


# 1. Define the architecture you trained
class MyChangeDetectionModel(nn.Module):
    def __init__(self):
        super(MyChangeDetectionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, imgA, imgB):
        x = torch.cat((imgA, imgB), dim=1)  # assumes input is [B, 3, H, W]
        return self.encoder(x)


# 2. Load the model weights
def load_model(model_path):
    model = MyChangeDetectionModel().cuda()
    state_dict = torch.load(model_path, map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# 3. Load and preprocess an image
def load_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # or your actual model input size
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).cuda()  # shape: [1, 3, H, W]
    return tensor


# 4. Predict using the model
def predict(model, imageA, imageB):
    model.eval()
    with torch.no_grad():
        output = model(imageA, imageB)
    return output


# 5. Plot result
def plot_predictions(predictions, imageA, imageB):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Image A")
    plt.imshow(imageA[0].permute(1, 2, 0).cpu().numpy())

    plt.subplot(1, 3, 2)
    plt.title("Image B")
    plt.imshow(imageB[0].permute(1, 2, 0).cpu().numpy())

    plt.subplot(1, 3, 3)
    plt.title("Predicted Change Map")
    plt.imshow(predictions[0][0].cpu().numpy(), cmap="hot")

    plt.tight_layout()
    plt.show()


# 6. Main entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict changes between two images using a trained model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=model_path,
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "--imageA", type=str, required=True, help="Path to the first input image."
    )
    parser.add_argument(
        "--imageB", type=str, required=True, help="Path to the second input image."
    )
    args = parser.parse_args()

    model = load_model(args.model_path)
    imageA = load_image(args.imageA)
    imageB = load_image(args.imageB)

    predictions = predict(model, imageA, imageB)
    plot_predictions(predictions, imageA, imageB)
