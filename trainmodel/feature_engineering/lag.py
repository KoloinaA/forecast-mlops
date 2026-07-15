import pandas as pd


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:

    group = ["store_nbr", "family"]

    lags = [1, 7, 14, 28]

    for lag in lags:

        df[f"sales_lag_{lag}"] = (
            df
            .groupby(group)["sales"]
            .shift(lag)
        )

    return df