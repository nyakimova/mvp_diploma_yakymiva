import pandas as pd
import numpy as np
import os

np.random.seed(42)

TARGET_ROWS = 2_500_000
CHUNK_SIZE = 100_000

OUTPUT_FILE = "data/raw/big_target_dataset.csv"
os.makedirs("data/raw", exist_ok=True)

SERVICE_CATEGORIES = [
    "Первинна медична допомога",
    "Спеціалізована амбулаторна допомога",
    "Стаціонарна медична допомога",
    "Хірургічна медична допомога",
    "Екстрена медична допомога",
    "Медична реабілітація",
    "Онкологічна допомога",
    "Психіатрична допомога",
    "Паліативна медична допомога"
]

generated = 0
first = True

while generated < TARGET_ROWS:
    rows = []

    for _ in range(CHUNK_SIZE):
        service_category = np.random.choice(SERVICE_CATEGORIES)

        avg_cost = np.random.lognormal(mean=9, sigma=0.7)
        coverage = np.clip(np.random.normal(0.9, 0.15), 0.5, 1.1)
        volume = np.clip(np.random.lognormal(4, 1.0), 5, 3000)

        cost_variance = np.random.beta(2, 5)
        payment_delay = np.random.exponential(30)
        complications_rate = np.random.beta(2, 10)

        reimbursement_rate = avg_cost * coverage

        deficit = (avg_cost - reimbursement_rate) / avg_cost
        instability = cost_variance + complications_rate + payment_delay / 60
        concentration = 1 - volume / 3000

        risk_score = (
            0.5 * deficit +
            0.3 * instability +
            0.2 * concentration +
            np.random.normal(0, 0.07)
        )

        if risk_score < 0.28:
            risk_category = "Низький фінансовий ризик"
        elif risk_score < 0.55:
            risk_category = "Середній фінансовий ризик"
        else:
            risk_category = "Високий фінансовий ризик"

        rows.append([
            service_category,
            avg_cost,
            reimbursement_rate,
            volume,
            cost_variance,
            payment_delay,
            complications_rate,
            risk_score,
            risk_category
        ])

    df = pd.DataFrame(rows, columns=[
        "service_category",
        "avg_cost",
        "reimbursement_rate",
        "volume_per_month",
        "cost_variance",
        "payment_delay",
        "complications_rate",
        "risk_score",
        "risk_category"
    ])

    df.to_csv(OUTPUT_FILE, mode="a", index=False, header=first)
    first = False
    generated += CHUNK_SIZE

    print(f"Згенеровано {generated:,} рядків")

print("big_target_dataset.csv створено")
