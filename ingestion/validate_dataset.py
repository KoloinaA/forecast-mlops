import pandas as pd
from config.config import RAW_DATA

df = pd.read_csv(RAW_DATA / "train.csv")

print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())