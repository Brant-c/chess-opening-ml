# For viewing data for testing

import pandas as pd

df = pd.read_parquet("data/processed/lichess_clean_2016_08.parquet")

print(df.columns.tolist())

print(df.head())
