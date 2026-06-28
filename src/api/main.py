# FASTAPI SERVICE - ECG HEARTBEAT ANALYSIS (Docker-ready version)

# Key change vs. local main file:
# models are now loaded from plain .pt files in the models/ folder, instead of from the MLflow Model Registry 
# the .pt files were created by "export_model.py". Everything else (endpoints, request/response schemas, inference logic) is IDENTICAL.
# also: using proper package-relative imports instead of sys.path trick
# -> now work reliabily in both local and Docker environments bc python treats src/ as a proper package (due to __init__.py files)
# -> "src.training.models" is a fully qualified module path (= no naming conflicts anymore with other similar packages/ names)


import os
import json
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
# import mlflow.pytorch - not needed anymore in the docker version, only for local version to load models
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

# Clean, fully qualified import: python looks for a package called "src" inside is a sub-package called "training", inside is a module called "models"
from src.training.models import ECGCNNClassifier, ECGAutoencoder

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
# now loading from .pt files at startup
# here "__file__" is the absolute path of THIS file (main.py) - is navigated here from the project root, then into models/.
# Works correctly both locally and inside Docker container (where /app is the WORKDIR and models/ is at /app/models/)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

print("Loading CNN classifier from .pt file...")
cnn_model = ECGCNNClassifier(num_classes=5) # creates empty model with same architecture as trained - rn with random weights - no learning
cnn_model.load_state_dict(torch.load(os.path.join(_MODELS_DIR, "cnn_model.pt"), map_location="cpu"))
cnn_model.eval()

print("Loading Autoencoder from .pt file...")
autoencoder_model = ECGAutoencoder(input_dim=187, bottleneck_dim=16)
autoencoder_model.load_state_dict(torch.load(os.path.join(_MODELS_DIR, "autoencoder_model.pt"), map_location="cpu"))
autoencoder_model.eval()

print("Loading anomaly threshold config...")
with open(os.path.join(_MODELS_DIR, "autoencoder_config.json")) as f:
    _config = json.load(f)
ANOMALY_THRESHOLD = _config["anomaly_threshold"]

print(f"All models loaded. Anomaly threshold = {ANOMALY_THRESHOLD:.6f}")



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