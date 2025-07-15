# 🛰️ Satellite Image Change Detection

A PyTorch-based deep learning pipeline for **semantic change detection** in high-resolution satellite imagery using the **LEVIR-CD** dataset.

---

## 📁 Dataset

**LEVIR-CD** (Change Detection Dataset for Remote Sensing Images)  
- [📥 Download from Kaggle](https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd)  
- Contains high-resolution pre-change and post-change image pairs with binary change masks.

🗂️ Dataset loader implementation: `datasetLevir.py`

---

## 🧠 Model Architecture

### 🔹 Siamese U-Net
- Implemented in `siameseUnetModel.py`
- A dual-branch encoder-decoder architecture that extracts features from **pre-event** and **post-event** images separately and then fuses them for final change prediction.
- Based on the U-Net design, with shared weights and skip connections.

---

## 🏋️‍♂️ Training

- Training script: `train_model.py`
- Optimizer: `AdamW`
- Learning Rate: Uses a **scheduler** for dynamic adjustment
- Loss Function:
  - Multiple losses tested
  - Best results achieved with **Focal-Tversky Loss**

📈 Training history is saved in: `training_history.pkl`

---

## 🔍 Inference

- Run predictions using: `test_model.py`
- Produces change masks from test image pairs using the trained Siamese U-Net model.

---

## 📦 File Overview

| File / Directory       | Description                                      |
|------------------------|--------------------------------------------------|
| `siameseUnetModel.py`  | Siamese U-Net model implementation               |
| `train_model.py`       | Model training loop                              |
| `test_model.py`        | Model inference / testing                        |
| `datasetLevir.py`      | Custom PyTorch `Dataset` class for LEVIR-CD     |
| `model_predit.py`      | (Alternative) prediction script                  |
| `metrics.py`           | Evaluation metrics                               |
| `best_model.pth`       | Saved best model weights                         |
| `training_history.pkl` | Training loss/accuracy history                   |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/arjunsingh4545/satelliteImageChangeDetection.git
cd satelliteImageChangeDetection

# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py

# main.py contains:
# - A training loop (from train_model.py)
# - A prediction function (from test_model.py)

# Modify main.py as per your use case (train, test, evaluate, etc.)
