import pandas as pd


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:

    group = ["store_nbr", "family"]

    windows = [7, 14, 30]

    for window in windows:

        df[f"rolling_mean_{window}"] = (
            df
            .groupby(group)["sales"]
            .transform(
                lambda x: x.shift(1).rolling(window).mean()
            )
        )

    df["rolling_std_7"] = (
        df
        .groupby(group)["sales"]
        .transform(
            lambda x: x.shift(1).rolling(7).std()
        )
    )

    return df