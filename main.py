from datasetLevir import train_loader, val_loader, test_loader
from siameseUnetModel import SiameseUNet
from train_model import train_model
from test_model import model_predict, plot_predictions
import torch
import time

if __name__ == "__main__":
    """
    start_time = time.time()
    model = SiameseUNet(in_channels=3, out_channels=1).to('cuda' if torch.cuda.is_available() else 'cpu')

    model = train_model(model, train_loader, val_loader, num_epochs=150, learning_rate=0.001)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    """
    # Load the pre-trained model
    model = SiameseUNet(in_channels=3, out_channels=1).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )
    )
    # Test the model

    start_time = time.time()
    predictions, imageA_list, imageB_list, masks_list = model_predict(
        model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    plot_predictions(predictions, imageA_list, imageB_list, masks_list, num_images=5)
    end_time = time.time()
    print(f"Testing completed in {end_time - start_time:.2f} seconds.")
    print("All operations completed successfully.")
