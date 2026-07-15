import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des variables calendaires.
    """

    df["year"] = df["date"].dt.year

    df["month"] = df["date"].dt.month

    df["day"] = df["date"].dt.day

    df["day_of_week"] = df["date"].dt.dayofweek

    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    df["quarter"] = df["date"].dt.quarter

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df