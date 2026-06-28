# MODEL EXPORT SCRIPT

# Important terms and definitions:
# Dockerfile: a text file with step by step instructions on how to build a docker image - each row generates a new layer in the image
# Base Image: the initial starting image - here python:3.11-slim is used - an official slim python image
# Layer Caching: Docker is caching each layer of the image - if the layer is not changing, it will be reused on the next docker build - makes rebuiling faster
               # Therefore: Dependencies (pip install) should be copied before the code (dependencies change rarely, code more often) - as a best practice
# .dockerignore: like gitignore - stops big folders/ files to be loaded into the image (venv/, data/, mlruns/, etc.)
# Port Mapping: "-p" - each container has its own network - using "-p 8000:8000" connects port 8000 of the container with port 8000 of the local machine (my pc) 
               # without this, noone from outside would be able to access the service
# WORKDIR: sets the "current directory" within the countainer for all following orders
# COPY: copy data from the host (my pc) to the image
# CMD: executing CMD to start a container out of the image


# Purpose of this script:
# Load both models from MLflow Model Registry and save them as standalone PyTorch .pt files in the models/ folder

# Why?
# MLflow is within .gitignore bc it is large, always changing and machine-specific
# -> .pt files in the models/ are tracked in git and can therefore be copied into the docker image at build time
# + makes API simpler -> just loading torch.load("models/cnn_model.pt")

# Background on torch.save() / torch.load()
# torch.save() serializes a Python object (here: the models state_dict -> being the learned weights, not the class definition) to a binary .pt file
# torch.load() deserializes it back. To reconstruct a full, usable model from a state_dict alone, the MODEL CLASS DEFINITION (ECGCNNClassifier, ECGAutoencoder) are needed also
# saving ONLY the state_dict (not the whole model object) is PyTorch's recommended practice: its more portable across PyTorch versions and doesnt depend 
# on pickle finding the class at a specific import path 


import sys
import os
import json
import torch
import mlflow.pytorch

# same sys.path trick as in main.py -> add src/training/ to Python's module search path to import the custom model classes (ECGCNNClassifier, etc.)
# which mlflow needs to reconstruct the models when loading them from the registry
_TRAINING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.append(_TRAINING_DIR)

# these imports must come AFTER sys.path.append
from models import ECGCNNClassifier, ECGAutoencoder


def main():
    # Export CNN Classifier
    print("Loading CNN Classifier from Model Registry...")
    cnn_model = mlflow.pytorch.load_model("models:/ecg-cnn-classifier@production")
    cnn_model.eval()
    torch.save(cnn_model.state_dict(), "models/cnn_model.pt")
    print("Saved models/cnn_model.pt")

    # Export Autoencoder
    print("Loading CNN Autoencoder from Model Registry...")
    autoencoder_model = mlflow.pytorch.load_model("models:/ecg-autoencoder@production")
    autoencoder_model.eval()
    torch.save(autoencoder_model.state_dict(), "models/autoencoder_model.pt")
    print("Saved models/autoencoder_model.pt")

    # Verify the anomaly threshold config exists (autoencoder_config.json was created by register_models.py - confirm its here bc its needed for API)
    if os.path.exists("models/autoencoder_config.json"):
        with open("models/autoencoder_config.json") as f:
            config = json.load(f)
        print(f"Anomaly threshold config found: {config}")
    else:
        print("WARNING: models/autoencoder_config.json not found!")
        print("Run src/training/register_models.py first.")

    print("\nExport complete. models/ folder contents:")
    for f in os.listdir("models"):
        size_kb = os.path.getsize(f"models/{f}") / 1024
        print(f"  {f}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()