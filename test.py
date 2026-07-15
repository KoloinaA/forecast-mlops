from trainmodel.training.loader import load_training_data
from trainmodel.training.split import temporal_split


df = load_training_data()


X_train, X_test, y_train, y_test = temporal_split(df)


print(X_train.shape)
print(X_test.shape)


print(y_train.shape)
print(y_test.shape)
