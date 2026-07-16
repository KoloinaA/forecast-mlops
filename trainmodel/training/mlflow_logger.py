import mlflow
import mlflow.sklearn

from contextlib import contextmanager
from mlflow.models import infer_signature


MLFLOW_TRACKING_URI = "http://mlflow:5000"



def setup_mlflow():
    
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    print(
        "MLflow URI:",
        mlflow.get_tracking_uri()
    )

    mlflow.set_experiment(
        "sales-forecast-v0.01"
    )



@contextmanager
def start_run(model_name):

    with mlflow.start_run(
        run_name=f"{model_name}_baseline"
    ):

        mlflow.log_param(
            "model",
            model_name
        )

        yield



def log_params(model):

    params = model.get_params()

    mlflow.log_params(
        params
    )



def log_metrics(metrics):

    for name, value in metrics.items():

        mlflow.log_metric(
            name,
            value
        )



def log_model(model, X_sample):
    
    print("Run actif :", mlflow.active_run())

    predictions = model.predict(X_sample)

    signature = infer_signature(
        X_sample,
        predictions
    )

    info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature
    )

    print(info)
    print("MODEL ENREGISTRE !")