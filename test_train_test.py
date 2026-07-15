# from trainmodel.training.loader import load_training_data
# from trainmodel.training.split import temporal_split


# df = load_training_data()


# X_train, X_test, y_train, y_test = temporal_split(df)


# print(X_train.shape)
# print(X_test.shape)
# print(X_train.dtypes)

# print(y_train.shape)
# print(y_test.shape)

from trainmodel.training.loader import load_training_data
from trainmodel.training.split import temporal_split
from trainmodel.training.preprocessing_model import create_preprocessor


df = load_training_data()


X_train, X_test, y_train, y_test = temporal_split(df)


preprocessor = create_preprocessor()


X_train_ready = preprocessor.fit_transform(X_train)


print(X_train_ready.shape)
