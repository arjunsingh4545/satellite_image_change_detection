#!/bin/bash

# full training pipeline script
# from downloading data to training the model

set -e  # Exit immediately if a command exits with a non-zero status

# initialising conda environment
ENV_NAME=myenv
PYTHON_VERSION=3.12

# Create the conda environment
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Initialize conda for bash (if not already done)
source ~/miniconda3/etc/profile.d/conda.sh  # or ~/anaconda3/...

# Activate the environment
conda activate $ENV_NAME

# Confirm activation
echo "Activated Conda environment: $ENV_NAME"

# Install required packages
pip install -r requirements.txt
# Check if the requirements were installed successfully
if [ $? -ne 0 ]; then
    echo "Failed to install required packages. Exiting."
    exit 1
fi

# Download the dataset
python download_dataset.py #installs via kaggle API. Change the script if you want to use a different method

# Check if the dataset was downloaded successfully
if [ $? -ne 0 ]; then
    echo "Failed to download dataset. Exiting."
    exit 1
fi

# Train the model
python train_model.py

# Check if the model was trained successfully
if [ $? -ne 0 ]; then
    echo "Failed to train the model. Exiting."
    exit 1
fi

