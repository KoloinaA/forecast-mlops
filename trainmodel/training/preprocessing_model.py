import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def create_preprocessor():

    categorical_features = [
        "family"
    ]


    numerical_features = [
        "store_nbr",
        "onpromotion",
        "year",
        "month",
        "day",
        "day_of_week",
        "week_of_year",
        "quarter",
        "is_weekend",
        "sales_lag_1",
        "sales_lag_7",
        "sales_lag_14",
        "sales_lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_30",
        "rolling_std_7"
    ]


    preprocessor = ColumnTransformer(

        transformers=[

            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore" #ignore les nouvelles valeurs de faily dans le futur, sans cela le modele plante en voyant une nouvelle valeur
                ),
                categorical_features
            ),

            (
                "num",
                "passthrough",
                numerical_features
            )
        ]
    )


    return preprocessor