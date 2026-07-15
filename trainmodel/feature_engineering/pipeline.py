import logging

from trainmodel.feature_engineering.loader import load_processed_data
from trainmodel.feature_engineering.calendar import add_calendar_features
from trainmodel.feature_engineering.lag import add_lag_features
from trainmodel.feature_engineering.rolling import add_rolling_features
from trainmodel.feature_engineering.saver import save_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def feature_engineering_pipeline():

    logger.info("Lecture des données...")

    df = load_processed_data()

    logger.info("Ajout des variables calendrier...")

    df = add_calendar_features(df)

    logger.info("Ajout des variables retard...")

    df = add_lag_features(df)

    logger.info("Ajout des rolling features...")

    df = add_rolling_features(df)

    logger.info("Suppression des lignes incomplètes dues aux lags...")

    df = df.dropna().reset_index(drop=True)

    logger.info(f"Dataset final : {df.shape}")

    save_features(df)

    logger.info("Feature Engineering terminé.")


if __name__ == "__main__":

    feature_engineering_pipeline()