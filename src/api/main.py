# FASTAPI SERVICE - ECG HEARTBEAT ANALYSIS

# this API provides 1 main endpoint, /predict, which takes a 187-value ECG signal and returns:
    # 1. a CNN-based classification (which of the 5 heartbeat types is most likely, and the model's confidence for each type)
    # 2. an Autoencoder-based anomaly score (does this ECG signal look "normal"?)

# Both models are loaded ONCE at startup (not per-request -> loading a model is relatively slow, running inference on an already-loaded model is fast)

import sys
import os
import json
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import mlflow.pytorch
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator


# important: making the custom model classes importable!

# CNN and AE were saved via mlflow.pytorch.log_model() (from inside src/training/train_cnn & *_autoencoder.py)
# there the model classes were imported like "from model import ..." -> possible bc "models" was automatically on sys.path 

# mlflow.pytorch.log_model() saves the model using pyhtons 'pickle' -> does not store a class's full source code 
# only stores reference like "models.ECGCNNClassifier" and expects python to be able to import models and find that class inside it when loading model later

# Within src/api/main.py -> src/training is not automatically on sys.path -> manually add
# (Cleaner long-term fix: restructuring src/training into a proper importable package with consistens import paths - maybe a future improvement)
_TRAINING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.append(_TRAINING_DIR)


# Class names
CLASS_NAMES = [
    "Normal (N)",                               # label 0
    "Supraventricular Premature (S)",           # label 1
    "Premature Ventricular Contraction (V)",    # label 2
    "Fusion of Ventricular and Normal (F)",     # label 3
    "Unclassifiable Beat (Q)",                  # label 4
]



# Create FastAPI application object
    # "title/ description/ version" show up automatically in the auto-generated /docs page -> nice free documentation for anyone using this API
app = FastAPI(
    title="ECG Heartbeat Analysis API",
    description="Classifies ECG heartbeats (5 classes, via CNN) and flags anomalous signals via reconstruction error (Autoencoder).",
    version="1.0.0",
)

# Load both models + anomaly threshold ONCE at module import time (i.e. when server starts, before it accepts requests)
print("Loading CNN classifier from MLflow Model Registry...")
cnn_model = mlflow.pytorch.load_model("models:/ecg-cnn-classifier@production")
cnn_model.eval()

print("Loading Autoencoder from MLflow Model Registry...")
autoencoder_model = mlflow.pytorch.load_model("models:/ecg-autoencoder@production")
autoencoder_model.eval()

print("Loading Autoencoder Anomaly Threshold...")
with open("models/autoencoder_config.json") as f:
    _autoencoder_config = json.load(f)
ANOMALY_THRESHOLD = _autoencoder_config["anomaly_threshold"]

print(f"Models loaded successfully. Anomaly threshold = {ANOMALY_THRESHOLD:.6f}")


# Request Schema:  /predict request body 
# "BaseModel" (from Pydantic) is FastAPI's way of describing expected JSON structure
# If request doesnt match this shape -> FastAPI automatically responds with clear validation error (HTTP 422)

class ECGSignal(BaseModel):
    # List[float]: a JSON array of numbers
    # Field(..., description=...): the "..." means this field is REQUIRED (no default value) - description shows up in the auto-generated /docs page
    signal: List[float] = Field(
        ...,
        description="187 ECG signal values, each scaled to the range [0, 1]"
    )

    # @field_validator: a custom check that runs AFTER Pydantic's built-in type checking -> after confirming that "signal" is a list of floats
    # @classmethod: is required by Pydantic v2's validator syntax

    # the following code enforces a constraint specific to the models that Pydantic's basic types cannot express (both expect exactly 187 values)
    # and if this raises a ValueError, FastAPI automatically turns it into a client-friendly HTTP 422 error
    @field_validator("signal")
    @classmethod
    def check_length(cls, value):
        if len(value) != 187:
            raise ValueError(f"Signal must contain exactly 187 values, get {len(value)}")
        return value
    

# Response Schema: /predict response body
# Defining this -> FastAPI validates MY OWN output too, and documents the exact response shape in /docs

class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_class_name: str
    class_probabilities: Dict[str, float]       # Dict[str,float] -> a JSON object mapping class names to probabilities
    anomaly_score: float
    anomaly_threshold: float
    is_anomaly: bool


# Endpoint 1: "health check"
# @app.get("health"): registers this function to handle HTTP GET requests to the /health URL
# health check endpoint is standard practice - tools like Kubernetes or load balancers can call this regularly to check:
# "is this service alive and responsive?" WITHOUT running expensive model inference.
@app.get("health")
def health():
    return {"status": "ok"}

# Endpoint 2: the main prediction endpoint.
# app.post("/predict", responde_model=PredictionResponse):
    # - POST: the client SENDS data (ECG signal) in the request body
    # response_model=PredictionResponse: FastAPI validates our return value against this schema, and documents it in /docs. 

# parameter "ecg: ECGSignal" tells FastAPI: "parse incoming JSON request body as ECGSignal object" -> running all its validations 
# -> and give it to me as 'ecg' 
@app.post("/predict", response_model=PredictionResponse)
def predict(ecg: ECGSignal):
    signal_array = np.array(ecg.signal, dtype=np.float32)   # Convert validated Python list of floats into numpy array 


    # PART 1: CNN classification
    # first .unsqueeze(0) -> adds a "batch" dimension -> shape (1, 187) -> 2. .unsqueeze(0) -> adds a "channel" dimension -> shape (1, 1, 187)
    # -> matches waht Conv1d expects: (batch, channels, length)
    cnn_input = torch.from_numpy(signal_array).unsqueeze(0).unsqueeze(0)

    # torch.no_graD() -> bc only referencing is done here, no need to track gradients
    with torch.no_grad():
        logits = cnn_model(cnn_input)   # shape (1, 5) -> raw scores
        probabilities = F.softmax(logits, dim=1).squeeze(0)   # convert raw scores into probabilities that sum to 1 (across dim=1 -> class dimension) -> shape (5,)
        predicted_class = int(torch.argmax(probabilities).item())   # softmax is order-preserving -> takes class with highest probability ==> Predicted class

    
    # Building a dictionary mapping each class NAME to its probability, rounded to 4 decimal places for a cleaner API response
    # round(float(probabilities[i]), 4): item()/float() converts a PyTorch scalar tensor to a plain Python float (JSON-serializable)
    class_probabilities = {
        CLASS_NAMES[i]: round(float(probabilities[i]), 4)
        for i in range(len(CLASS_NAMES))
    }


    # PART 2: Autoencoder
    # linear layers -> expect (batch_size, 187)
    ae_input = torch.from_numpy(signal_array).unsqueeze(0)

    with torch.no_grad():
        reconstruction = autoencoder_model(ae_input)
        anomaly_score = ((ae_input - reconstruction) ** 2).mean().item()    # MeanSq difference between input and reconstruction, item() 
                                                                            # -> converts single-element tensor to plain python float
        
    # Assemble the response
    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_class_name=CLASS_NAMES[predicted_class],
        class_probabilities=class_probabilities,
        anomaly_score=round(anomaly_score, 6),
        anomaly_threshold=round(ANOMALY_THRESHOLD, 6),
        is_anomaly=anomaly_score > ANOMALY_THRESHOLD,
    )


