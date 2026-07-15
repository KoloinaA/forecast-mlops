from config.config import PROCESSED_DATA


def save_features(df):

    output = PROCESSED_DATA / "train_features.csv"

    df.to_csv(output, index=False)

    print(f"Features sauvegardées dans {output}")