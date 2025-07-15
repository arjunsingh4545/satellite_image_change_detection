import torch
from metrics import iou_metric_pytorch, FocalTverskyLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='cuda',
                l2_regularisation=0.0001, PATIENCE=10 , CHECKPOINT_PATH='best_model.pth'):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_regularisation)
    criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6)  # Focal Tversky Loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Monitor a "minimum" quantity (validation loss)
        factor=0.5,  # Reduce LR by 50%
        patience=5,  # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=1e-7  # Minimum learning rate
    )

    history = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': [],
        'iou_metric': [],  # <- renamed from 'iou'
        'val_iou_metric': []  # <- renamed from 'val_iou'
    }

    best_val_loss = float('inf')
    epochs_no_improvement = 0

    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_accuracy = 0.0
        num_train_batches = 0

        for (imageB, imageA), mask in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training"):
            imageB, imageA, mask = imageB.to(device), imageA.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(imageB, imageA)
            loss = criterion(mask.unsqueeze(1), outputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            outputs = torch.sigmoid(outputs)  # Apply sigmoid activation to get probabilities
            train_iou += iou_metric_pytorch(mask, outputs).item()
            correct = ((outputs > 0.5).float() == mask.unsqueeze(1)).float()
            train_accuracy += correct.sum().item() / correct.numel()
            num_train_batches += 1

        avg_train_loss = train_loss / num_train_batches
        avg_train_iou = train_iou / num_train_batches
        avg_train_accuracy = train_accuracy / num_train_batches  # Average accuracy over batches

        # Store training metrics in history
        history['loss'].append(avg_train_loss)
        history['iou_metric'].append(avg_train_iou)
        history['accuracy'].append(avg_train_accuracy)

        model.eval()  # Set model to evaluation mode (disables dropout, batchnorm updates)
        val_loss = 0.0
        val_iou = 0.0
        val_accuracy = 0.0
        num_val_batches = 0
        with torch.no_grad():  # Disable gradient computation for validation to save memory and speed up
            for (images_b, images_a), masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation"):
                # Move data to device
                images_b, images_a, masks = images_b.to(device), images_a.to(device), masks.to(device)

                # Forward pass
                outputs = model(images_b, images_a)
                loss = criterion(masks.unsqueeze(1), outputs)

                # Accumulate metrics
                val_loss += loss.item()
                outputs = torch.sigmoid(outputs)  # Apply sigmoid activation to get probabilities
                val_iou += iou_metric_pytorch(masks, outputs).item()
                correct = ((outputs > 0.5).float() == masks.unsqueeze(1)).float()
                val_accuracy += correct.sum().item() / correct.numel()
                num_val_batches += 1

                # Calculate average validation metrics for the epoch
        avg_val_loss = val_loss / num_val_batches
        avg_val_iou = val_iou / num_val_batches
        avg_val_accuracy = val_accuracy / num_val_batches

        # Store validation metrics in history
        history['val_loss'].append(avg_val_loss)
        history['val_iou_metric'].append(avg_val_iou)
        history['val_accuracy'].append(avg_val_accuracy)

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, Train Acc: {avg_train_accuracy:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}, Val Acc: {avg_val_accuracy:.4f}")

        scheduler.step(avg_val_loss)
        # --- Callbacks: Checkpoint and Early Stopping ---
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
            best_val_loss = avg_val_loss
            # Save only the model's state dictionary (weights and biases)
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            epochs_no_improve = 0  # Reset patience counter
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Patience: {epochs_no_improve}/{PATIENCE}")
            if epochs_no_improve == PATIENCE:
                print("Early stopping triggered! Restoring best model weights...")
                # Load the weights from the best epoch saved earlier
                model.load_state_dict(torch.load(CHECKPOINT_PATH))
                break  # Exit the training loop

    print("Training complete.")
    # Save training history to disk
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print("Training history saved to training_history.pkl")

    # --- Plotting Training History ---
    print("\n--- Plotting Training Metrics ---")

    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(history['iou_metric'], label='Training IoU')
    plt.plot(history['val_iou_metric'], label='Validation IoU')
    plt.title('Training and Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
import pickle
import matplotlib.pyplot as plt

def plot_training_history(pickle_path):
    # Load training history
    with open(pickle_path, 'rb') as f:
        history = pickle.load(f)

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot IoU
    plt.figure(figsize=(10, 6))
    plt.plot(history['iou_metric'], label='Training IoU')
    plt.plot(history['val_iou_metric'], label='Validation IoU')
    plt.title('Training and Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

