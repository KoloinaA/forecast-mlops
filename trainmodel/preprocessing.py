import logging
from pathlib import Path

import pandas as pd

from config.config import RAW_DATA, PROCESSED_DATA

# ----------------------------
# Configuration du logging
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ----------------------------
# Colonnes attendues
# ----------------------------

EXPECTED_COLUMNS = [
    "date",
    "store_nbr",
    "family",
    "sales",
    "onpromotion",
]


# ----------------------------
# Chargement
# ----------------------------

def load_data(filename: str = "train.csv") -> pd.DataFrame:
    filepath = RAW_DATA / filename

    logger.info(f"Lecture du fichier : {filepath}")

    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} introuvable.")

    df = pd.read_csv(filepath)

    logger.info(f"{len(df)} lignes chargées.")

    return df


# ----------------------------
# Validation des colonnes
# ----------------------------

def validate_columns(df: pd.DataFrame):

    logger.info("Validation des colonnes...")

    missing = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Colonnes manquantes : {missing}"
        )

    logger.info("Toutes les colonnes sont présentes.")


# ----------------------------
# Informations générales
# ----------------------------

def report_dataset(df: pd.DataFrame):

    logger.info(f"Dimensions : {df.shape}")

    logger.info("\nTypes des colonnes :")
    logger.info(df.dtypes)

    logger.info("\nValeurs manquantes :")
    logger.info(df.isna().sum())


# ----------------------------
# Suppression des doublons
# ----------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    logger.info(f"Doublons supprimés : {before-after}")

    return df


# ----------------------------
# Conversion des dates
# ----------------------------

def convert_dates(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Conversion de la colonne date...")

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        f"Période : {df['date'].min()} -> {df['date'].max()}"
    )

    return df


# ----------------------------
# Sauvegarde
# ----------------------------

def save_data(df: pd.DataFrame):

    output = PROCESSED_DATA / "train_processed.csv"

    df.to_csv(output, index=False)

    logger.info(f"Fichier sauvegardé : {output}")


# ----------------------------
# Pipeline principal
# ----------------------------

def preprocessing_pipeline():

    logger.info("=" * 60)
    logger.info("Début du preprocessing")
    logger.info("=" * 60)

    df = load_data()

    validate_columns(df)

    report_dataset(df)

    df = remove_duplicates(df)

    df = convert_dates(df)

    save_data(df)

    logger.info("=" * 60)
    logger.info("Preprocessing terminé avec succès")
    logger.info("=" * 60)


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    preprocessing_pipeline()