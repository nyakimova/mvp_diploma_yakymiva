import pandas as pd
import os

INPUT_FILE = "data/raw/big_target_dataset.csv"
OUTPUT_FILE = "data/processed/train_sample.csv"

os.makedirs("data/processed", exist_ok=True)

SAMPLE_ROWS = 300_000
CHUNK_SIZE = 100_000

samples = []
current_rows = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
    remaining = SAMPLE_ROWS - current_rows
    if remaining <= 0:
        break

    take_n = min(len(chunk), remaining)
    sampled_chunk = chunk.sample(n=take_n, random_state=42)

    samples.append(sampled_chunk)
    current_rows += len(sampled_chunk)
    print(f"Набрано {current_rows} рядків")

df = pd.concat(samples, ignore_index=True)
df.to_csv(OUTPUT_FILE, index=False)

print("train_sample.csv створено")
print(df["risk_category"].value_counts(normalize=True))
