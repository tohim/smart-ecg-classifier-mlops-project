# AUTOENCODER TRAINING SCRIPT - ANOMALY DETECTION

# Trains the Autoencoder on normal ECG heartbeats only. Then use it to detect abnormal heartbeats based on reconstruction error.

# Overall structure/ flow: Training -> Threshold Calibration -> Final Evaluation

# 1. Training (similar pattern to CNN) - Training AE to minimize reconstruction error (MSE) on normal training data
    # track validation loss per epoch, keep best checkpoint

# 2. Threshold calibration: Use best Checkpoint, compute reconstruction errors on still normal-only validation set.
    # pick threshold such that most normal validation examples fall BELOW it (e.g., 95th percentile)
    # = 95% of normal validation examples have an error below this value, by construction

# 3. Final evaluation: Combine held-out normal test set AND the abnormal set into 1 evaluation set.
    # For each example -> compute reconstruction error and compare to threshold.
    # if error > threshold -> anomaly, otherwise -> normal.
    # Compare predictions to true labels (normal vs abnormal) - compute accuracy, precision, recall, F1 and ROC-AUC.


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report)
import mlflow
import mlflow.pytorch
from dataset import ECGAutoencoderDataset
from models import ECGAutoencoder


def compute_reconstruction_errors(model, X, device):

    # This function is a standalone helper, as it will be reused again in a later phase  
    # Given a trained AE + a np array of inputs X w shape (num_samples, 187) -> return a np array of shape (num_samples,)
    # -> containing the PER-SAMPLE reconstruction error (mean squared error between each input and its reconstruction)

    model.eval()    # here we dont have a dropout layer, still good practice

    with torch.no_grad():

        X_tensor = torch.from_numpy(X).to(device)   # convert to tensor and move to device (dataset with few thousand rows x 187 columns 
                                                    # -> fits comfortably in memory, no need to process in batches)

        reconstructions = model(X_tensor)           # run full batch through AE to get reconstruction (what gets returned by forward function in the model)
                                                    # with shape (num_samples, 187)

        errors = ((X_tensor - reconstructions) ** 2).mean(dim=1)    # elementwise squared difference, mean(dim=1) -> avg across the 187 columns = 1 number per sample

    return errors.cpu().numpy() # move back to cpu and convert to numpy for use with numpy/scikit-learn functions later




def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load all 4 preprocessed arrays 
    X_train = np.load("data/processed/ae_X_train.npy")
    X_val = np.load("data/processed/ae_X_val.npy")
    X_test_normal = np.load("data/processed/ae_X_test_normal.npy")
    X_test_abnormal = np.load("data/processed/ae_X_test_abnormal.npy")

    print(f"Train (normal only): {X_train.shape}")
    print(f"Val (normal only):   {X_val.shape}")
    print(f"Test normal:         {X_test_normal.shape}")
    print(f"Test abnormal:       {X_test_abnormal.shape}")

    # Part 1: Training Setup

    # only X_train and X_val are wrapped in Dataset/DataLoader - no iteration over the test sets in batches -> they are handled later via compute_reconstruction_errors()
    train_dataset = ECGAutoencoderDataset(X_train)
    val_dataset = ECGAutoencoderDataset(X_val)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = 187
    bottleneck_dim = 16
    model = ECGAutoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim).to(device)

    # nn.MSELoss() -> Mean Squared Error -> for autoencoders this is standard loss
    # Measures: how different is the reconstruction from the input?  -> should be minimized turing training + serves as anomaly score
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 50 epochs: our training set here is much smaller than the ECG classification dataset, so each epoch is very fast -> can take more 
    # after first run: val_loss was decreasing until epoch 50 and not yet converged
    # new try is with 100 epochs, also important: will this influence the ROC-AUC? As: lower reconstruction error on NORMAL data doesnt automatically 
    # mean better separation from abnormal data.
    num_epochs = 100

    
    # Best-checkpoint tracking: here tracking the LOWEST val_loss (lower = better, as it is an error measure - so we initialize with +infinity)
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state = None

    mlflow.set_experiment("ecg-anomaly-detection")

    with mlflow.start_run(run_name="autoencoder_ptbdb"):
        mlflow.log_param("model_type", "Dense Autoencoder")
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("bottleneck_dim", bottleneck_dim)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("loss_function", "MSE")
        mlflow.log_param("checkpoint_strategy", "best val_loss")

        # Training Loop.
        # Same 4 step cycle as in CNN -> forward, compute loss, backward, optimizer step.
        # only now: the target that is compared is the input itself - and not a separate label

        for epoch in range(num_epochs):

            model.train()
            running_train_loss = 0.0

            for signals in train_loader:

                signals = signals.to(device)
                optimizer.zero_grad()
                
                reconstructions = model(signals)            # Forward Pass
                loss = criterion(reconstructions, signals)  # Loss
                loss.backward()                             # Backward Pass
                optimizer.step()                            # Optimizing

                running_train_loss += loss.item() * signals.size(0)

            train_loss = running_train_loss / len(train_dataset)

            # Validation Phase
            model.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for signals in val_loader:
                    signals = signals.to(device)
                    reconstructions = model(signals)
                    loss = criterion(reconstructions, signals)
                    running_val_loss += loss.item() * signals.size(0)

                val_loss = running_val_loss / len(val_dataset)

                print(
                    f"Epoch {epoch + 1} / {num_epochs} - "
                    f"train_loss: {train_loss:.6f} - "
                    f"val_loss: {val_loss:.6f}"
                )

                mlflow.log_metric("epoch_train_loss", train_loss, step=epoch)
                mlflow.log_metric("epoch_val_loss", val_loss, step=epoch)

                # Checkpointing - keep model with lowest val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    best_model_state = {
                        key: value.clone() for key, value in model.state_dict().items()
                    }

                    print(f"  -> New best model (val_loss={val_loss:.6f}), checkpoint saved.")

        # Restore best checkpoint before doing anything else
        model.load_state_dict(best_model_state)
        print(f"\nBest epoch: {best_epoch}/{num_epochs} (val_loss={best_val_loss:.6f})")


        # Part 2: Threshold Calibration
        # compute reconstruction error for val set (normal heartbeats the model did NOT train on, but also did NOT influence its final weight selection)

        val_errors = compute_reconstruction_errors(model, X_val, device)

        # np.percentile(array, 95) -> finds the values below which 95% of the arrays values fall.
        # -> 5% of NORMAL validation examples will have an error above it -> i.e. 5% "false alarm rate" on data we know is normal
        # goal: to reach a reasonable trade-off between sensitivity and false alarms. 
        # therefore, 95 is a reasonable starting point

        threshold_percentile = 95
        threshold = np.percentile(val_errors, threshold_percentile)

        print(
            f"\nReconstruction error threshold "
            f"({threshold_percentile}th percentile of val errors): {threshold:.6f}"
        )


        # Part 3: Final Evaluation

        # compute reconstruction errors for BOTH held-out test sets
        test_normal_errors = compute_reconstruction_errors(model, X_test_normal, device)
        test_abnormal_errors = compute_reconstruction_errors(model, X_test_abnormal, device)

        # Quick sanity check: are abnormal errors on average higher than normal errors? 
        print(
            f"\nReconstruction error of test NORMAL: "
            f"mean={test_normal_errors.mean():.6f}, std={test_normal_errors.std():.6f}"
        )
                
        print(
            f"\nReconstruction error of test ABNORMAL: "
            f"mean={test_abnormal_errors.mean():.6f}, std={test_abnormal_errors.std():.6f}"
        )

        # Combine both test sets into one big evalation set using np.concatenate -> joins 2 arrays end-to-end into one longer array
        all_errors = np.concatenate([test_normal_errors, test_abnormal_errors])

        # Build the TRUE labels for this combined set: 0=Normal, 1=anomaly
        y_true = np.concatenate([
            np.zeros(len(test_normal_errors)),
            np.ones(len(test_abnormal_errors))
        ])

        # Apply the threshold
        y_pred = (all_errors > threshold).astype(int)   # creates boolean array -> .astype(int) converts true/false to 1/0  
                                                        # -> y_pred[i] = 1 -> prediction is "anomaly" -> based purely on wheter its reconstruction error exceeds 
                                                        # the calibrated threshold 

        # Threshold dependent metrics (using y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # Threshold independent metric: ROC-AUC
        # does not use y_pred (the thresholded 0/1 predictions)
        # Instead: directly uses continuous scores (all_errors) together with the true labels (y_true)
        # Asks: If I pick a random anomaly and a random normal example, how often does the anomaly get a HIGHER error score than the NORMAL one? 
        # "1" means -> always perfect separation | "0.5" -> no better than random guessing
        # Tells: How good is the reconstruction error as a signal in general, separate from specific threshold 

        roc_auc = roc_auc_score(y_true, all_errors)

        print(f"\nAnomaly detection results (threshold={threshold:.6f}):")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")

        print("\nDetailed report:")
        print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))


        # Log everything to MLflow

        # Add additional Threshold Tade-Off Exploration.
        # 95th was picked as central threshold - however, other thresholds might deliver better results.
        # To broaden the perspective, the precision/recall/f1 for several percentile choices are computed based on existing values
        # Why? -> "right" threshold depends on operational priorities.
        # e.g. Medical Screening Context: missing abnormal best might be far more costly than a false alarm (low precision)
            # -> potentially better to choose a LOWER pctl than 95 -> accepting more false alarms
        # Threshold-Exploration Table should make this trade-off visible
        print("\nThreshold trade-off (using different percentiles of val errors):")
        print(f"{'Percentile':>10} | {'Threshold':>10} | {'Precision':>9} | {'Recall':>7} | {'F1':>6}")
        print("-" * 55)

        threshold_tradeoff = []
        for pct in [50, 75, 90, 95, 99]:
            t = np.percentile(val_errors, pct)
            y_pred_t = (all_errors > t).astype(int)
            p = precision_score(y_true, y_pred_t)
            r = recall_score(y_true, y_pred_t)
            f = f1_score(y_true, y_pred_t)

            print(f"{pct:>10} | {t:>10.6f} | {p:>9.4f} | {r:>7.4f} | {f:>6.4f}")

            threshold_tradeoff.append({
                "percentile": pct,
                "threshold": float(t),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
            })


        mlflow.log_metric("threshold", threshold) # threshold is logged as a metric (not param), since number is derived from data (val_errors) - it could change if retraining or calibrating
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("precision", float(precision))
        mlflow.log_metric("recall", float(recall))
        mlflow.log_metric("f1", float(f1))
        mlflow.log_metric("roc_auc", float(roc_auc))

        mlflow.log_param("best_epoch", best_epoch)
        mlflow.log_param("threshold_percentile", threshold_percentile)

        # log_dict() serialized the python dict/list as a JSON file and attaches it to this run
        mlflow.log_dict({"threshold_tradeoff": threshold_tradeoff}, "threshold_tradeoff.json")


        mlflow.pytorch.log_model(model, "model")

        print("\nRun complete. Start 'mlflow ui' to view the results.")
          

if __name__ == "__main__":
    main()


            

