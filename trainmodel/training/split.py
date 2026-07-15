import pandas as pd


def temporal_split(
        df: pd.DataFrame,
        target: str = "sales"
):

    # Toujours trier par date
    df = df.sort_values("date")


    # Nombre de dates uniques
    n_dates = df["date"].nunique()


    # Point de séparation
    split_point = int(n_dates * 0.8)


    dates = (
        df["date"]
        .drop_duplicates()
        .sort_values()
    )

    print("Dernière date train :")
    print(dates.iloc[split_point-1])


    print("\nPremière date test :")
    print(dates.iloc[split_point])
    train_dates = dates.iloc[:split_point]

    test_dates = dates.iloc[split_point:]


    train = df[
        df["date"].isin(train_dates)
    ]


    test = df[
        df["date"].isin(test_dates)
    ]


    # Séparation X / y

    X_train = train.drop(
        columns=[target, "date"]
    )

    y_train = train[target]


    X_test = test.drop(
        columns=[target, "date"]
    )

    y_test = test[target]


    return (
        X_train,
        X_test,
        y_train,
        y_test
    )