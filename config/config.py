from pathlib import Path

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# Dossiers de données
RAW_DATA = BASE_DIR / "data" / "raw"
PROCESSED_DATA = BASE_DIR / "data" / "processed"

# Création automatique des dossiers
RAW_DATA.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA.mkdir(parents=True, exist_ok=True)