import torch

# =============================
# Intersection over Union (IoU)
# =============================
def iou_metric_pytorch(y_true, y_pred):
    """
    Compute the IoU (Jaccard Index) between prediction and ground truth masks.
    Inputs:
        y_true: Ground truth mask (tensor)
        y_pred: Model prediction (tensor)
    Returns:
        IoU score as a scalar tensor
    """
    y_pred = (y_pred > 0.5).float()  # Binarize predictions
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)  # Avoid division by zero

# =============================
# Dice Loss
# =============================
def dice_loss_pytorch(y_true, y_pred):
    """
    Compute Dice loss (1 - Dice coefficient).
    Inputs:
        y_true: Ground truth mask
        y_pred: Predicted mask (probability values between 0 and 1)
    Returns:
        Dice loss as a scalar tensor
    """
    y_pred = y_pred.view(-1)  # Flatten predictions
    y_true = y_true.view(-1)  # Flatten labels
    intersection = (y_pred * y_true).sum()
    dice = (2.0 * intersection + 1e-15) / (y_pred.sum() + y_true.sum() + 1e-15)
    return 1 - dice  # Dice loss

# ========================================
# Combined BCE + Dice Loss (weighted BCE)
# ========================================
def bce_dice_loss_pytorch(y_true, y_pred):
    """
    Compute a combination of Binary Cross Entropy and Dice Loss.
    Inputs:
        y_true: Ground truth mask
        y_pred: Raw logits from the model
    Returns:
        Combined loss (BCE + Dice)
    """
    # Binary Cross Entropy with class imbalance handling (pos_weight > 1 favors positives)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        y_pred,
        y_true,
        reduction="mean",
        pos_weight=torch.tensor([5.0], device=y_pred.device)  # Adjust if class imbalance changes
    )

    # Apply sigmoid to logits for dice calculation
    y_pred_sigmoid = torch.sigmoid(y_pred)
    dice = dice_loss_pytorch(y_true, y_pred_sigmoid)

    return bce + dice  # Total loss

# ========================================
# Precision and Recall Metrics
# ========================================
def precision_recall(preds, labels, threshold=0.5):
    """
    Compute precision and recall for binary predictions.
    Inputs:
        preds: Predicted probabilities or logits
        labels: Ground truth binary mask
        threshold: Threshold to binarize predictions
    Returns:
        Tuple: (precision, recall)
    """
    preds = (preds > threshold).float()  # Binarize predictions
    labels = labels.float()

    TP = torch.sum(preds * labels)
    FP = torch.sum(preds * (1 - labels))
    FN = torch.sum((1 - preds) * labels)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    return precision.item(), recall.item()

# ========================================
# Focal Tversky Loss (for highly imbalanced segmentation)
# ========================================
class FocalTverskyLoss(torch.nn.Module):
    """
    Focal Tversky Loss:
    - Tversky index is a generalization of Dice score with tunable weights for FP/FN.
    - Focal term focuses on hard-to-classify examples.
    Suitable for class-imbalanced problems.
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, smooth=1e-6):
        """
        alpha: weight for false positives
        beta: weight for false negatives
        gamma: focal parameter to penalize easy examples less
        smooth: for numerical stability
        """
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, targets, inputs):
        """
        Inputs:
            targets: ground truth mask
            inputs: raw logits from the model
        Returns:
            Focal Tversky Loss value
        """
        inputs = torch.sigmoid(inputs)  # Convert logits to probabilities
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positive (TP), False Positive (FP), False Negative (FN)
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        # Tversky index formula
        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        # Focal Tversky loss
        loss = torch.pow((1 - tversky_index), self.gamma)
        return loss
