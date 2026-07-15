import pandas as pd

from config.config import PROCESSED_DATA


def load_training_data():

    path = PROCESSED_DATA / "train_features.csv"

    df = pd.read_csv(
        path,
        parse_dates=["date"]
    )

    return df

