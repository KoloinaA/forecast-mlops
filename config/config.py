from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = BASE_DIR / "data" / "raw"

PROCESSED_DATA = BASE_DIR / "data" / "processed"

EXTERNAL_DATA = BASE_DIR / "data" / "external"

VALIDATION_DATA = BASE_DIR / "data" / "validation"