import pandas as pd

from config.config import PROCESSED_DATA


def load_processed_data():

    path = PROCESSED_DATA / "train_processed.csv"

    return pd.read_csv(path, parse_dates=["date"])