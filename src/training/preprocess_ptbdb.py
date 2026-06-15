# PTBDB PREPROCESSING SCRIPT (for Autoencoder-based Anomaly Detection)

# Initial Autoencoder related definitions:
# 1. Autoencoder:
    # A NN consisting of 2 major parts: Encoder + Decoder. It learns to reconstruct its own input.
# 2. Encoder:
    # Compresses the input (here the 187 values) step by step into smaller representations
# 3. Decoder:
    # Tries to reconstruct (decode) the original full resolution input based on the compressed representation from the encoder
# 4. Bottleneck/ Latent Space: smallest, most compressed representation between encoder and decoder (here chosen 16 instead of 187 values)
    # the NN MUST still keep the most important information, everything else is lost 
# 5. Reconstruction Error: Measures how strong the input is different from the reconstructed output -> central anomaly metric
# 6. MSE (Mean Squared Error): Is calculated as the average of the squared differences between 2 rows of values.
    # Serves as the loss function (for training) and additional reconstruction-error-metric (for evaluation)
    # it is "what the NN is trained on" basically. And also serves as the final "Anomaly-Score" of the Autoencoder
# 7. Self-supervised Learning: the model is learning the "correct" answer out of the data itself - no external true labels needed/ given
# 8. Threshold: A cut-off value for the reconstruction error -> once reached, a signal is categorized as "anomaly". Is usually set as a percentile of the errors
    # given within the validation data (e.g., 95. percentile = the value under which 95% of all normal examples are)
# 9. ROC-AUC (Area Under the ROC Curve): is a metric between 0 and 1. Measures how good a score is (here: the reconstruction error) DIFFERENTIATING between 2 classes
    # its an independent measure of a concrete threshold (0.5 = random chance, 1 = perfect)
    # useful bc it tells: "how good is the score differentiating generally?" -> before finally deciding on a distinct and set threshold
# 10. Sigmoid-Activation: an activation function which squeezes each value between 0 and 1. It is the last layer of the decoder (necessary, as the ECG values in the dataset
    # are also already normalized from 0 to 1) -> Input and Output should have the same range of values - only then the MSE will be reasonable 


# Script Information:
# Loading the PTB Diagnostics ECG data and prepare it for training/ evaluating an autoencoder
# Unlike preprocessing.py, this script does NOT need to keep track of class labels for training - the autoencoder only ever sees "normal" heartbeats during training
# and learns to reconstruct them.

# 4 separate arrays, with distinct roles are needed:
# 1. ae_X_train:         normal heartbeats, model adjusts weights based on these
# 2. ae_X_val:           separate set of normal heartbeats, used for: (a) tracking validation loss during training (to pick best checkpoint) 
#                        and (b) AFTER training: calibrating the anomaly threshold 
# 3. ae_X_test_normal:   a 3rd set of normal heartbeats, held out from both training and threshold calibration. For final evaluation.
# 4. ae_X_test_abnormal: ALL abnormal heartbeats. Final evaluation: Does model correctly flag these as anomalies?


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split    # a scikit-learner helper that randomly splits array into 2 parts
                                                        # will be called twice in a row to get three-way split (70%/ 15%/ 15%)


def main():

    # Step 1: Load both CSV files
    normal_df = pd.read_csv("data/raw/ptbdb_normal.csv", header=None)
    abnormal_df = pd.read_csv("data/raw/ptbdb_abnormal.csv", header=None)

    # Quick data-validation check: each file represents 1 class - label column should be constant within each file - 0-normal, 1-abnormal"
    print("Unique label values in normal file:   ", normal_df.iloc[:,-1].unique())
    print("Unique label values in abnormal file: ", abnormal_df.iloc[:,-1].unique())

    # Step 2: Extract just the signal values (drop the label column)
    X_normal = normal_df.iloc[:, :-1].values.astype("float32")
    X_abnormal =abnormal_df.iloc[:, :-1].values.astype("float32")

    # Check
    print(f"\nNormal Heartbeats:   {X_normal.shape}")
    print(f"\nAbnormal Heartbeats: {X_abnormal.shape}")

    # Step 3: Three-way split of NORMAL
    X_normal_train, X_normal_rest = train_test_split(X_normal, test_size=0.30, random_state=42)     # splits 70% for training and rest 
    X_normal_val, X_normal_test = train_test_split(X_normal_rest, test_size=0.50, random_state=42)  # splits the rest 30% into 2 * 15% sets

    # Check
    print(f"\nNormal train (for AE training):          {X_normal_train.shape}")
    print(f"Normal val (for checkpoints + threshold): {X_normal_val.shape}")
    print(f"Normal test (final evaluation only):      {X_normal_test.shape}")
    print(f"Abnormal (final evaluation only):         {X_abnormal.shape}")

    # Save everything
    np.save("data/processed/ae_X_train.npy", X_normal_train)
    np.save("data/processed/ae_X_val.npy", X_normal_val)
    np.save("data/processed/ae_X_test_normal.npy", X_normal_test)
    np.save("data/processed/ae_X_test_abnormal.npy", X_abnormal)

    print("\nDone. Files saved in data/processed/")

if __name__ == "__main__":
    main()