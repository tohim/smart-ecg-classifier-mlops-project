# MODEL REGISTRATION SCRIPT

# Definitions:

# 1. Model Registry: central collection of models and their names. Every registry/ registration produces a new version.
# 2. Model Alias: a flexibel pointer to a specific version (e.g., "production" -> "Version 3"). Code always refernces the alias (e.g., models:/ecg-cnn-classifier@production)
                # - and never a specific version - after retraining everything is just moved to the alias, without changing the code
# 3. REST API: interface accessed via HTTP - "GET" to pull, "POST" to send data (here the ECG signal)
# 4. FastAPI: Python-Framework, used to build the API
# 5. Pydantic-Model: a class describing request/ response 
# 6. Uvicorn: Server that is actually run by the FastAPI-App and listening to a port
# 7. Swagger UI/ "/docs": FastAPI automatically generates an interactive web surface under http://localhost:8000/docs, to directly test endpoints in the browser
# 8. Softmax: Transforms raw CNN outpus (logits) into percentages (values between 0-1, sum up to 1 total). Until now only used argmax (to get class with highest value)
                # for API it will also be useful to additionally report the confidence


# Script Information
# Find the best CNN run + best Autoencoder run (based on the metrics)
# Register each of them in MLflow's Model Registry under a stable name with a "production" alias.
# Reason: if API says "load the model from run x" -> would need to manually find and hardcode that ID + update it everytime we retrain, etc.
# -> instead use STABLE NAMES ("ecg-cnn-classifier", "ecg-autoencoder") and an alias "production"
# -> this always points at "whichever version we currently consider the one to use"
# -> API will simply ask for "models:/ecg-cnn-classifier@production" (without need to know about run IDs at all)

# The script automates the "which was best run?" decision by querying MLflow directly for the run with the HIGHEST value of a chosen metric
# f1_macro for CNN - roc_auc for the autoencoder

import json
import mlflow
from mlflow.tracking import MlflowClient

def register_best_run(experiment_name, metric_name, registry_name):

    # find the run with highest metric_name inside experiment_name, register model under register_name in Model Registry + point a "production" alias at this new version
    # function then returns the MLflow Run object for the best run (so caller can read additional metrics)

    # MLflowClient is a lower-level interface than the 'mlflow.*' -> gives access to search/registry operations not available as simple top-level functions
    client = MlflowClient()

    # look up experiment 
    experiment = client.get_experiment_by_name(experiment_name)

    # search_runs() queries all runs in this experiment ; order_by=[f"metrics.{metrics_name} DESC"] sorts so run with highest value of metric comes first 
    # for both f1_macro and roc_auc higher is better
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{metric_name} >= 0",
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=1,
    )

    best_run = runs[0]
    metric_value = best_run.data.metrics[metric_name]
    print(
        f"Best run in '{experiment_name}': "
        f"run_id={best_run.info.run_id}, "
        f"{metric_name}={metric_value:.4f}"
    )

    # URI format "runs:/<run_id>/model" refers to the "model" artifact folder of a specific run
    # -> same that is used in mlflow.pytorch.log_model(model, "model") during training
    model_uri = f"runs:/{best_run.info.run_id}/model"

    # mlflow.register_model(): -> if "registry_name" doesnt exist -> creates new version under that name -> pointing at model_uri
    # returns a ModelVersion object; ".version" is the new version number as a string
    result = mlflow.register_model(model_uri, registry_name)
    print(f"Registered '{registry_name}' version {result.version}")

    # set_registered_model_alias(): -> points the alias "production" at this specific version
    # in case "production" was pointing elsewhere (e.g. earlier version from previous run) -> it gets moved to new version
    # -> "moves the pointer after retraining"-case
    client.set_registered_model_alias(registry_name, "production", result.version)
    print(f"Alias 'production' -> '{registry_name}' version {result.version}")

    return best_run



def main():
    # CNN Classifier -> pick highest f1_macro across all runs in this experiment 
    register_best_run(
        experiment_name="ecg-heartbeat-classification",
        metric_name="f1_macro",
        registry_name="ecg-cnn-classifier",
    )

    print()

    # Autoencoder -> pick highest roc_auc
    # also, saving the returning varaible in "autoencoder_run" -> later needed to gather more metrics 
    autoencoder_run = register_best_run(
        experiment_name="ecg-anomaly-detection",
        metric_name="roc_auc",
        registry_name="ecg-autoencoder"
    )
    
    # save the AE anomaly threshold for the API (threshold value was logged as metric during training, but its just a number - not part of the model)
    # therefore not included when register/load model - but it was saved as a small json config file that API will read at startup
    threshold = autoencoder_run.data.metrics["threshold"]

    config = {"anomaly_threshold": threshold}
    with open("models/autoencoder_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved anomaly threshold ({threshold:.6f}) to models/autoencoder_config.json")


if __name__ == "__main__":
    main()