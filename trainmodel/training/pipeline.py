from trainmodel.training.loader import load_training_data
from trainmodel.training.split import temporal_split

from trainmodel.training.preprocessing_model import create_preprocessor

from trainmodel.training.models import get_models

from trainmodel.training.trainer import (
    train_model,
    create_training_pipeline
)

from trainmodel.training.evaluator import evaluate_model

from trainmodel.training.mlflow_logger import (
    setup_mlflow,
    start_run,
    log_metrics,
    log_model,
    log_params
)



def training_pipeline():


    print("Initialisation MLflow...")

    setup_mlflow()


    print("Chargement des données...")

    df = load_training_data()



    print("Split temporel...")

    X_train, X_test, y_train, y_test = temporal_split(df)



    print("Création du préprocesseur...")

    preprocessor = create_preprocessor()



    models = get_models()

    results = {}



    for name, model in models.items():


        print(f"Entraînement : {name}")


        with start_run(name):


            pipeline_model = create_training_pipeline(
                preprocessor,
                model
            )


            trained_model = train_model(
                pipeline_model,
                X_train,
                y_train
            )


            metrics = evaluate_model(
                trained_model,
                X_test,
                y_test
            )


            print(metrics)


            # Enregistrement MLflow

            log_metrics(metrics)


            log_params(
                model
            )


            log_model(
                trained_model,
                X_test.head(5),
            )


            results[name] = metrics



        



    return results



if __name__ == "__main__":


    results = training_pipeline()

    print(results)