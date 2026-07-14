from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://admin:admin@postgres:5432/forecast_db"
)

with engine.connect() as conn:
    print("Connexion réussie à PostgreSQL !")