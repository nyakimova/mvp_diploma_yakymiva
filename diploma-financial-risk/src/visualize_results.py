import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams["figure.dpi"] = 130

df = pd.read_csv("data/processed/train_sample.csv")

model = joblib.load("models/catboost_model.pkl")
le = joblib.load("models/label_encoder.pkl")

X_model = df.drop(columns=["risk_category", "risk_score"])
y = le.transform(df["risk_category"])

risk_map = {
    "Низький фінансовий ризик": "Низький",
    "Середній фінансовий ризик": "Середній",
    "Високий фінансовий ризик": "Високий"
}

df_vis = df.copy()
df_vis["risk_short"] = df_vis["risk_category"].map(risk_map)

df_vis["coverage_ratio"] = df_vis["reimbursement_rate"] / df_vis["avg_cost"]
df_vis["cost_per_case"] = df_vis["avg_cost"] / df_vis["volume_per_month"]
df_vis["delay_pressure"] = df_vis["payment_delay"] / 60
df_vis["instability_proxy"] = (
    df_vis["cost_variance"] + df_vis["complications_rate"]
)

plt.figure(figsize=(6, 4))
sns.countplot(
    data=df_vis,
    x="risk_short",
    order=["Низький", "Середній", "Високий"],
    palette=["#6CBF84", "#F4C430", "#E06666"]
)
plt.title("Розподіл рівнів фінансового ризику", fontsize=14)
plt.xlabel("Рівень ризику")
plt.ylabel("Кількість контрактів")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(
    data=df_vis,
    x="risk_score",
    hue="risk_short",
    bins=40,
    kde=True,
    stat="density",
    element="step"
)
plt.title("Розподіл інтегрального показника фінансового ризику", fontsize=14)
plt.xlabel("Risk score")
plt.ylabel("Щільність")
plt.tight_layout()
plt.show()

corr_features = [
    "avg_cost",
    "coverage_ratio",
    "volume_per_month",
    "cost_per_case",
    "delay_pressure",
    "instability_proxy",
    "risk_score"
]

corr = df_vis[corr_features].corr()

# прибираємо числовий шум
corr[np.abs(corr) < 0.01] = 0

plt.figure(figsize=(9, 7))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title("Кореляція фінансових та операційних факторів", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

probs = model.predict_proba(X_model)

high_label = "Високий фінансовий ризик"
high_idx = le.transform([high_label])[0]

y_bin = (y == high_idx).astype(int)

precision, recall, _ = precision_recall_curve(
    y_bin,
    probs[:, high_idx]
)

pr_auc = average_precision_score(
    y_bin,
    probs[:, high_idx]
)

plt.figure(figsize=(6, 5))
plt.plot(
    recall,
    precision,
    linewidth=2,
    label=f"PR-AUC = {pr_auc:.3f}"
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR-крива для класу «Високий фінансовий ризик»", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

preds = model.predict(X_model).astype(int)
cm = confusion_matrix(y, preds)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Низький", "Середній", "Високий"],
    yticklabels=["Низький", "Середній", "Високий"],
    linewidths=0.5
)
plt.xlabel("Прогнозований клас")
plt.ylabel("Фактичний клас")
plt.title("Матриця помилок класифікатора CatBoost", fontsize=14)
plt.tight_layout()
plt.show()
