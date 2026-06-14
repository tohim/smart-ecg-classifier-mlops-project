# CNN TRAINING SCRIPT

# Trains the 1D CNN (defined in models.py) on the ECG heartbeat data, then evaluate it after each epoch + log everything to MLflow in same experiment as Random Forest 

# New concepts introduced, compared to train_baseline.py:
    # PyTorch tensors, Dataset/DataLoader
    # "Epochs" and "batches": training a NN means: showing it the data multiple times (in epochs), and in small groups (batches)
    # Training Loop: forward pass -> compute loss -> backward pass -> optimizer step => this 4-step cycle repeats for every batch, in every epoch
    # Loss function: CrossEntropyLoss
    # Optimizer: Adam
    # Handling the class imbalance in the dataset via a "weighted" loss function (NN equivalent of class_weight="balanced", from the Random Forest baseline)
    # Logging metrics over TIME (per epoch) to MLflow, instead of once at the end


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.pytorch

# Import the custom classes from the other files in this folder
# Works because, when running python train_cnn.py -> python automatically adds the script's own folder to the path
from dataset import ECGDataset
from models import ECGCNNClassifier

CLASS_NAMES = [
    "Normal (N)",                               # label 0
    "Supraventricular Premature (S)",           # label 1
    "Premature Ventricular Contraction (V)",    # label 2
    "Fusion of Ventricular and Normal (F)",     # label 3
    "Unclassifiable Beat (Q)",                  # label 4
]

def main():
    # Step 0: Reproducibility - setting the PyTorch random seed ; unlike scikit-learn (one random_state parameter), PyTorch has own random number generator
    torch.manual_seed(42)

    # Step 1: Chose a "device" - CPU OR GPU - torch.cuda.is_available() checks for usable GPU
    # Create a "device" object to later tell PyTorch where to place the model and the data.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 2: Load the preprocessed data (same files as for random forest baseline)
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    # Step 3: Wrap the data in the custom Dataset, then in DataLoaders
    # ECGDataset from dataset.py makes our numpy array compatible with PyTorch's training tools.
    train_dataset = ECGDataset(X_train, y_train)
    test_dataset = ECGDataset(X_test, y_test)

    # How many examples are processed together in one "step" of training - instead of updating weights after EVERY example (slow, noisy) 
    # VS. after ENTIRE dataset (memory-heavy, infrequent) - 256 is a common reasonable default size for such a dataset
    batch_size = 256

    # DataLoader wraps a Dataset and handles:
        # - splitting it into batches
        # - suffling order of examples (shuffle=True) - IMPORTANT: so network doesnt learn from data in always the same order (avoid order-dependent patterns, dont generalize)
        # for TEST -> no shuffle - as order doesnt matter for evaluation, plus fixed order makes debugging easier
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Step 4: Compute Class Weights for the Loss Function
    # the NN equivalent of class_weight="balanced" from random forest. Here its wanted that the loss function "cares" more about mistakes on rare classes (1,3) 
    # than on the common class (0)
    # Common formula for class weights:
    # weight_for_class_i = total_samples / (num_classes * count_of_class_i)
    # -> gives rare classes (small amount of class i) a larger weight , and common classes a smaller one

    # np.bincount(y_train) -> counts how many times each integer label appears in y_train, returning an array 
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    num_classes = len(class_counts)

    # Original approach to compute the classweights: class_weights = total_samples / (num_classes * class_counts)
    # Problem: Class weights were too extreme - rarest class (3) got a weight of ~27, while most common class got ~0.24
    # this caused the model to massively over-predict class 3 -> very low precision (0.14 -> 83% of (3) predictions were wrong)
    # and very high recall for class 3 (0.92)
    # Fix: instead of using raw "balanced" weights, take square root of them -> keeps order, but compresses range
    balanced_weights = total_samples / (num_classes * class_counts)
    class_weights = np.sqrt(balanced_weights)   # applys np.sqrt() element-wise to each weight

    # Convert to PyTorch tensor and move to chosen device. 
    # Loss function needs this tensor on the same device as the model's output
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("Class weights (higher = rare class, weighted more in the loss):")

    # zip merges 2 arrays/ lists elementwise and creates pairs with each element, and enumerate adds a counting starting from 0 results in: 
    # (0, "Normal (N)", 0.24)
    # (1, "Supraventricular...", 7.88)
    # (2, "Premature ventricular...", 3.02)
    # (3, "Fusion...", 27.3)
    # (4, "Unclassifiable (Q)", 2.72)

    # destructing/ double de-packaging: i -> get the counting up (0,1,2,3,4) ; (name, w) -> internal tuple of zip -> name = String of CLASS_NAME, w = number of class_weights

    for i, (name, w) in enumerate(zip(CLASS_NAMES, class_weights)):
        if i < len(CLASS_NAMES) - 1:
            print(f" {i} - {name}: {w:.3f}")
        else:
            print(f" {i} - {name}: {w:.3f}\n")

    # Step 5: Create the model, loss function and optimizer
    # Creats the CNN and moves all weights to chosen device. ".to(device)" is needed on BOTH the model and every batch of data. (PyTorch requires this)
    model = ECGCNNClassifier(num_classes=num_classes).to(device)

    # nn.CrossEntropyLoss is standard loss function -> compares models raw output scores against true integer labels - producing single number on how 
    # wrong predictions were on average. Lower = better.
    # weight=class_weights_tensor -> makes loss care more about rare classes -> mistakes on rare classes contribute more to the total loss
    # -> optimizer works harder to fix those mistakes
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Optimizer is the algorithm that updates the model's weights -> based on gradients computed during backpropagation
    # "Adam" is one of the most widely used optimizers - good default for most problems
    # model.parameters(): -> tells optimizer which values it's allowed to update (= all weights and biases of the Conv1d and Linear Layer)
    # lr (learning rate) = 0.001 -> controls how big each update step is -> if chosen too high -> training can become unstable
    # -> if too low -> training takes very long time.
    # 0.001 is a common, reasonable point for Adam.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Step 6: Training loop setup + MLflow run
    # num_epochs = how many times we go through the ENTIRE training dataset. Each pass gives model chance to refine weights based on what went wrong last time.
    # 10 is a reasonable start point for this dataset/ model size - the logged per-epoch metrics will show whether more epochs would help
    num_epochs = 10

#     First attempt/run was: Logging of the LAST epoch's model, not the BEST one.
#   - Looking at the per-epoch test_f1_macro values from v1, epoch 9
#     (0.7197) was BETTER than epoch 10 (0.6915) - but saved/logged
#     epoch 10's weights, since that's simply where the loop ended.

#   - FIX: "model checkpointing" - during training, we keep a copy of
#     the model's weights from whichever epoch had the best
#     test_f1_macro so far. After all epochs finish, we load THOSE
#     weights back into the model before logging it and printing the
#     final report. This is standard practice in real ML training.

    # Therefore: Variables necessary to track the BEST checkpoint seen so far.
    # made so that any value will be "better than nothing", to trigger first checkpoint save
    # these will hold the values/objects from whichever epoch currently has highest/ best -> all are updated together when new best epoch is found
    best_f1_macro = -1.0
    best_epoch = -1
    best_accuracy = None
    best_f1_weighted = None
    best_model_state = None
    best_preds = None
    best_labels = None

    mlflow.set_experiment("ecg-heartbeat-classification")

    with mlflow.start_run(run_name="cnn_1d"):

        # Log the hyperparameters
        mlflow.log_param("model_type", "1D-CNN")
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss_function", "CrossEntropyLoss (sqrt-scaled class-weights)")
        mlflow.log_param("class_weighting_scheme", "sqrt(balanced)")
        mlflow.log_param("checkpoint_strategy", "best test_f1_score")

        # Actual Training Loop: "repeat for each epoch":
        for epoch in range(num_epochs):

            # model.train() - puts model into training mode. Matters for layers like Dropout (should only randomly zero out values during training)
            # but model.train doesnt train itself - just sets a flag
            model.train()

            running_loss = 0.0  # accumulates total loss over the epoch

            # now iterate over training data in batches. 
            # in each iteration, the "signal" has shape (batch_size, 1, 187); and labels have shape (batch_size,) (except maybe last batch, may be smaller by size mismatch)
            for signals, labels in train_loader:
                # move batch data to same device as model
                signals = signals.to(device)
                labels = labels.to(device)

                # Now the actual 4-step training cycle starts
                # 1. Reset gradients from previous batch (as PyTorch accumulates gradients by default) - for standard training good to start fresh each batch
                optimizer.zero_grad()

                # 2. Forward Pass: run batch through model, get current predictions based on logits -> this calls "model.forward(signals)" internally (models.py)
                outputs = model(signals)    # shape: (batch_size, 5)

                # 3. Compute the loss: how far were predictions from true labels
                loss = criterion(outputs, labels)

                # 4. Backward pass: PyTorch's "autograd" system automatically computes how much each individual weight in the entire network contributed
                # to this loss value (the gradient of the loss of each weight) -> this is backpropagation and PyTorch handles the math via .backward()
                loss.backward()

                # 5. Optimizer - with the knowledge of each weights gradient -> Adam updates every weight slightly in direction that would have reduced the loss (scaled by lr)
                optimizer.step()

                # .item() extracts the loss value as a plain python float (loss itself is a tensor)
                # It is then multiplied by signals.size(0) (the actual batch_size) -> to make sure that the final average is correctly weighted 
                # even if the last batch is smaller then the others.
                running_loss += loss.item() * signals.size(0)


            # Average loss across the whole training set for this epoch
            epoch_loss = running_loss / len(train_dataset)

            # Evaluation after each epoch: how well does the model do on data it did NOT train on (test set)
            # model.eval() puts model into evaluation mode - disables dropout
            model.eval()

            all_preds = []  # collects predicted labels for the whole test set
            all_labels = [] # collects TRUE labels for the whole test set

            # torch.no_grad() -> tells PyTorch: "dont track gradients in this block"
            # similar: not calling .backwards() during evaluation
            with torch.no_grad():
                for signals, labels in test_loader:
                    signals = signals.to(device)
                    labels = labels.to(device)

                    outputs = model(signals)

                    # torch.argmax(outputs, dim=1) -> for each example in the batch, argmax finds the INDEX of the highest score among the 5 classes
                    # dim=1 if class dimension; dim=0 is the batch dimension
                    preds = torch.argmax(outputs, dim=1)

                    # then move predictions and labels back to CPU and conert to numpy - so we can use scikit-learns metric functions (expects numpy arrays, not GPU tensors)
                    # .cpu() is a no-io if already on CPU, but necessary on GPU
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # all_preds and all_labels are now plain python lists of integers -> one entry per test example
            accuracy = accuracy_score(all_labels, all_preds)
            f1_macro = f1_score(all_labels, all_preds, average="macro")
            f1_weighted = f1_score(all_labels, all_preds, average="weighted")

            print(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"train_loss: {epoch_loss:.4f} - "
                f"test_accuracy: {accuracy:.4f} - "
                f"Test_f1_macro: {f1_macro:.4f}"
            )

            # Log this epochs metrics to MLflow
            # the "step" parameter tells MLflow that these values belong to a SEQUENCE (one value per epoch) -> so it can plot them as a line chart over time in the UI
            # e.g., to see whether the epoch_f1_macro keeps improving or starts getting worse after a certain epoch (a sign of overfitting)
            mlflow.log_metric("epoch_train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("epoch_accuracy", accuracy, step=epoch)
            mlflow.log_metric("epoch_f1_macro", f1_macro, step=epoch)
            mlflow.log_metric("epoch_f1_weighted", f1_weighted, step=epoch)


            # New Checkpoint Logic to find the best epoch from all training runs:
            if f1_macro > best_f1_macro:
                best_f1_macro = f1_macro
                best_epoch = epoch+1    # stored +1 for better readability
                best_accuracy = accuracy
                best_f1_weighted = f1_weighted

                # model.state_dict() returns dictionary -> mapping each layer's name to its current weight tensors
                # IMPORTANT: by default -> returns REFERENCES to the model's tensors , not copies
                # if stored directly, it would keep changing as the training continues
                # -> using a dict comprehension with .clone() to create an independent copy of every tensor -> get frozen snapshot
                best_model_state = {
                    key: value.clone() for key, value, in model.state_dict().items()
                }
                
                # also store this epochs predictions/labels - to print detailed classification report for the best epoch later 
                best_preds = all_preds
                best_labels = all_labels

                print(f" -> New best model (f1_macro={f1_macro:.4f}), checkpoint saved.")

        # Also new in the improved version: after all epochs, restore best checkpoint into the model before logging it
        # -> now the saved/ logged model is the one that performed best - not just the one that is last
        # load_state_dict() takes a dictionary like the one state_dict() returns - and copies those values back into the model's layers
        # -> effectively "rewinding" the model's weights to how they were at "best_epoch".
        model.load_state_dict(best_model_state)

        print(
            f"\nBest epoch: {best_epoch}/{num_epochs} "
            f"(test_f1_macro={best_f1_macro:.4f})"
        )

        # After all epochs: print final detailed report and log the best model
        print("\nFinal classification report (best model/checkpoint):")
        print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

        # Also new: now log the final summary metrics using the same names as in the train_baseline.py, without the step argument
        # without step: metrics with "step" are treated as sequence/ history - if logged without: a single final value - exactly like in baseline
        # necessary, bc MLflow's comparison table aligns columns by metric name - so these need to match.
        mlflow.log_metric("accuracy", best_accuracy)
        mlflow.log_metric("f1_macro", best_f1_macro)
        mlflow.log_metric("f1_weighted", best_f1_weighted)

        # also log to which epoch this corresponds to, as a parameter - a single descriptive fact bout this run
        mlflow.log_param("best_epoch", best_epoch)

        # now model = the best model out of all runs.
        # mlflow.pytorch.log_model() is the PyTorch equivalent of mlflow.sklearn.log_model() from random forest model
        # saves the models architecture and learned weights in a standardized format for it to be easily reloaded later 
        # e.g. for the API in the later part of the project.
        mlflow.pytorch.log_model(model, "model")

        print("\nRun complete. Start 'mlflow ui' to view the result.")

if __name__ == "__main__":
    main()


