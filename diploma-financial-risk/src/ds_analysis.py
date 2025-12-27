import pandas as pd

df = pd.read_csv("data/processed/train_sample.csv")

print("Розмір датасету:", df.shape)
print("\nТипи даних:")
print(df.dtypes)

print("\nПропущені значення:")
print(df.isna().sum())

print("\nРозподіл класів:")
print(df["risk_category"].value_counts(normalize=True))

print("\nСтатистичний опис:")
print(df.describe())
