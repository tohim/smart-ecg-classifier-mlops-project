# BASELINE MODEL TRAINING SCRIPT

# Training a Random Forest classifier on the preprocessed ECG heartbeat dataset, evaluate it and log everyting 
# (parameters, metrics, trained model - using MLflow) -> Benchmark number to compare against Neural Network

# Background:
# Random Forest is a collection of many individual decision trees
# Each tree is trained on a slightly different random subset of the data and features
# For the final prediction, each tree votes for a class and the forest picks the class with the most votes (majority voting)

# MLflow: Experiment tracking tool, logs the used settings: parameters; model performance: metrics; and artifacts: trained model file, confusion matrix plot, etc.
# all organized in a web UI 

import numpy as np
from sklearn.ensemble import RandomForestClassifier                           # Random Forest Classifier model from scikit-learn
from sklearn.metrics import accuracy_score, f1_score,  classification_report  # functions to evaluate the model performance
import mlflow                                                                 # MLflow library for experiment tracking
import mlflow.sklearn                                                         # Sub-module for logging scikit-learn models



# same class names as in preprocessing.py, but we don't want to import the whole file just for this small list
CLASS_NAMES = [
    "Normal (N)",                               # label 0
    "Supraventricular Premature (S)",           # label 1
    "Premature Ventricular Contraction (V)",    # label 2
    "Fusion of Ventricular and Normal (F)",     # label 3
    "Unclassifiable Beat (Q)",                  # label 4
]


def main():

    # Step 1: Load the preprocessed data from disk using np.load() 
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy") 
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    # Step 2: Define Hyperparameters

    # Defome Hyperparameters (settings that control the training process, not learned from data - as opposed to "model parameters" 
    # like weights in a neural network, which the algorithm learns during training)

    # n_estimators: number of individual decision trees -> more trees, usually more stable, but more memory and longer training time
    n_estimators = 50

    # max_depth: maximum number of decision levels per tree, limiting prevents overfitting
    max_depth = 15

    # random_state: a seed for the random number generator, random forests use randomness internally, setting a seed makes the results reproducible
    random_state = 42

    # class_weight: tells Random Forest model how to handle class imbalance (~83% of samples are "Normal"), chosing "balanced" gives more 
    # importance to minority classes during training, which can improve performance on those classes
    class_weight = "balanced"

    # Step 3: Tell MLflow which "experiment" this run belongs to -> MLflow experiment is like a folder/ category grouping related runs together
    # Later, the CNN and the Autoencoder will log to the same experiment name, so all models show up together for comparison
    mlflow.set_experiment("ecg-heartbeat-classification")

    # Step 4: Start an MLflow run and train/ eval inside it
    # "with mlflow.start_run()" -> is a Python context manager -> everthing indented inside belongs to one tracked run
    # MLflow automatically:
        # - records start/end time of the run
        # - groups all params/ metrics/ artifacts we log inside this block under one unique run ID
        # - closes the run cleanly when the block ends (even on error)

    with mlflow.start_run(run_name = "random_forest_baseline"):

        # Log parameters: ".log_params(key, value)" -> saves 1 config value tagged with name
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("class_weight", class_weight)

        # Create and train the model
        print("Training in progress ... (may take 1-2 minutes)")

        # Create an "instance" of the Random Forest Classifier with the defined hyperparameters
        # at this point, the model is just an empty shell/ template, it hasn't learned anything yet

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1,                     # Performance setting: use all available CPU cores for faster training
        )

        # ".fit(X,y)" is the training step: trains to which class (label, y) each heartbeat (data, X) belongs to
        model.fit(X_train, y_train)


        # Make predictins on the test set using the trained model
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        # overall accuracy: how many heartbeats were classified correctly
        accuracy = accuracy_score(y_test, y_pred)     

        # macro-averaged F1 score: takes simple, unweighted average F1 score across all classes, treating them equally (good for imbalanced datasets)
        f1_macro = f1_score(y_test, y_pred, average="macro")   

        # weighted-averaged F1 score: calculates F1 score for each class and averages them, weighted by the number of true instances for each class (gives more importance to majority class)
        f1_weighted = f1_score(y_test, y_pred, average="weighted")   


        # Print results to console
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print(f"F1 Score (Weighted): {f1_weighted:.4f}")


        # classification_report() generates a detailed table showing for each class: precision, recall, F1 score, and support (number of true instances)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))


        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)


        # Save the trained model as an MLflow "artifact"
        # mlflow.sklearn.log_model(model, "model"):
            # 1. Serialized (saves) the trained model object to disk in a standard format (pickle file for scikit-learn models)
            # 2. Stores it inside this run's folder under the name "model"
            # 3. Records metadata about how to load this model later, this enables the "Model Registry" concept. MLflow can later reload this exact model
            #    without any needing to remember which scikit-learn version or code produced it.
        mlflow.sklearn.log_model(model, "model")

        print("\nRun complete. Start 'mlflow ui' to view the results.")

if __name__ == "__main__":
    main()
    