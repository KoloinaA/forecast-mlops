from trainmodel.training.loader import load_training_data
from trainmodel.training.split import temporal_split
from trainmodel.training.preprocessing_model import create_preprocessor
from trainmodel.training.models import get_models
from trainmodel.training.trainer import train_model
from trainmodel.training.evaluator import evaluate_model



def training_pipeline():


    print("Chargement des données...")

    df = load_training_data()



    print("Split temporel...")

    X_train, X_test, y_train, y_test = temporal_split(df)



    print("Préprocessing...")

    preprocessor = create_preprocessor()


    X_train_ready = preprocessor.fit_transform(
        X_train
    )

    X_test_ready = preprocessor.transform(
        X_test
    )



    models = get_models()



    results = {}



    for name, model in models.items():


        print(f"Entraînement : {name}")


        trained_model = train_model(
            model,
            X_train_ready,
            y_train
        )


        metrics = evaluate_model(
            trained_model,
            X_test_ready,
            y_test
        )


        results[name] = metrics



    return results



if __name__ == "__main__":

    results = training_pipeline()

    print(results)