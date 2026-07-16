from sklearn.pipeline import Pipeline



def create_training_pipeline(
        preprocessor,
        model
):

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )

        ]

    )


    return pipeline



def train_model(
        pipeline,
        X_train,
        y_train
):

    pipeline.fit(
        X_train,
        y_train
    )


    return pipeline