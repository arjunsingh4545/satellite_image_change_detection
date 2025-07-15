import torch


def iou_metric_pytorch(y_true, y_pred):
    # Binarize prediction for IoU calculation
    y_pred = (y_pred > 0.5).float()
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    # Add epsilon for numerical stability to avoid division by zero
    return (intersection + 1e-15) / (union + 1e-15)


def dice_loss_pytorch(y_true, y_pred):
    # Flatten tensors for easier calculation
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()
    # Add epsilon for numerical stability
    dice = (2.0 * intersection + 1e-15) / (y_pred.sum() + y_true.sum() + 1e-15)
    return 1 - dice


def bce_dice_loss_pytorch(y_true, y_pred):
    # Using functional API for BCE, ensuring `y_pred` and `y_true` are correctly shaped
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        y_pred,
        y_true,
        reduction="mean",
        pos_weight=torch.tensor([5.0], device=y_pred.device),
    )
    y_pred_sigmoid = torch.sigmoid(y_pred)
    dice = dice_loss_pytorch(y_true, y_pred_sigmoid)
    return bce + dice


def precision_recall(preds, labels, threshold=0.5):
    preds = (preds > threshold).float()
    labels = labels.float()

    TP = torch.sum(preds * labels)
    FP = torch.sum(preds * (1 - labels))
    FN = torch.sum((1 - preds) * labels)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    return precision.item(), recall.item()


class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, targets, inputs):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        loss = torch.pow((1 - tversky_index), self.gamma)
        return loss
