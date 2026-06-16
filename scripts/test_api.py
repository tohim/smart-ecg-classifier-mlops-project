# SIMPLE and QUICK API TEST SCRIPT

# Sends one real ECG signal per class (5 total) from the test set to the running API's /predict endpoint, and prints the response
# Script assumes that the API is already running 

import numpy as np
import requests

API_URL = "http://127.0.0.1:8000/predict"

CLASS_NAMES = [
    "Normal (N)",
    "Supraventricular premature (S)",
    "Premature ventricular contraction (V)",
    "Fusion ventricular/normal (F)",
    "Unclassifiable (Q)",
]


def main():
    # load CNN test data 
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    for true_class in range(5):
        idx = np.where(y_test == true_class)[0][0]  # returns indicies of all examples with this label; [0] takes the first such index
        signal = X_test[idx].tolist()   # (187,) - convert to plain python list so it can be sent to JSON (bc np arrays arent directly JSON-serializable)
        response = requests.post(API_URL, json={"signal": signal})  # sends HTTP POST request with given dict serialized as the JSON request body 
                                                                    #(what ECGSignal Pydantic model expects)
        result = response.json()    # .json() parses the JSON response body back into a Python dict

        print(f"\n--- True class: {true_class} ({CLASS_NAMES[true_class]}) ---")
        print(f"Predicted class: {result['predicted_class']} ({result['predicted_class_name']})")
        print(f"Class probabilities: {result['class_probabilities']}")
        print(f"Anomaly score: {result['anomaly_score']} (threshold: {result['anomaly_threshold']})")
        print(f"Is anomaly: {result['is_anomaly']}")


if __name__ == "__main__":
    main()

