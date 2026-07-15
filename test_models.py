from trainmodel.training.loader import load_training_data
from trainmodel.training.split import temporal_split
from trainmodel.training.preprocessing_model import create_preprocessor
from trainmodel.training.models import get_models
from trainmodel.training.trainer import train_model
from trainmodel.training.evaluator import evaluate_model


# Chargement
df = load_training_data()


# Split temporel
X_train, X_test, y_train, y_test = temporal_split(df)


# Préprocessing
preprocessor = create_preprocessor()


X_train_ready = preprocessor.fit_transform(
    X_train
)


X_test_ready = preprocessor.transform(
    X_test
)


# Modèles
models = get_models()


model = models["linear_regression"]


# Entraînement
trained_model = train_model(
    model,
    X_train_ready,
    y_train
)


# print(trained_model)
# print(
#     trained_model.coef_.shape
# )

metrics = evaluate_model(
    trained_model,
    X_test_ready,
    y_test
)


print(metrics)