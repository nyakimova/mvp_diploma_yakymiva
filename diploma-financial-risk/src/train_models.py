import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    average_precision_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

df = pd.read_csv("data/processed/train_sample.csv")

TARGET = "risk_category"
DROP_COLS = ["risk_score"] 

X = df.drop(columns=[TARGET] + DROP_COLS)
y = df[TARGET]


label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_enc,
    test_size=0.25,
    stratify=y_enc,
    random_state=42
)

X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_test_enc = pd.get_dummies(X_test, drop_first=True)

# вирівнюємо колонки
X_test_enc = X_test_enc.reindex(
    columns=X_train_enc.columns,
    fill_value=0
)


lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

lr.fit(X_train_enc, y_train)
lr_pred = lr.predict(X_test_enc)

lr_bal_acc = balanced_accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred, average="weighted")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=14,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train_enc, y_train)
rf_pred = rf.predict(X_test_enc)

rf_bal_acc = balanced_accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average="weighted")

cat_features = [X.columns.get_loc("service_category")]

cb = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.07,
    loss_function="MultiClass",
    eval_metric="TotalF1",
    verbose=False,
    random_seed=42
)

cb.fit(X_train, y_train, cat_features=cat_features)

cb_pred = cb.predict(X_test).astype(int)
cb_proba = cb.predict_proba(X_test)

cb_bal_acc = balanced_accuracy_score(y_test, cb_pred)
cb_f1 = f1_score(y_test, cb_pred, average="weighted")


high_label = "Високий фінансовий ризик"
high_idx = label_encoder.transform([high_label])[0]

y_test_binary = (y_test == high_idx).astype(int)

cb_pr_auc = average_precision_score(
    y_test_binary,
    cb_proba[:, high_idx]
)


metrics = pd.DataFrame({
    "Модель": [
        "Логістична регресія",
        "Випадковий ліс",
        "CatBoost"
    ],
    "Збалансована точність": [
        round(lr_bal_acc, 3),
        round(rf_bal_acc, 3),
        round(cb_bal_acc, 3)
    ],
    "F1-міра (зважена)": [
        round(lr_f1, 3),
        round(rf_f1, 3),
        round(cb_f1, 3)
    ],
    "PR-AUC (високий ризик)": [
        "—",
        "—",
        round(cb_pr_auc, 3)
    ]
})

print("\n" + "=" * 70)
print("ПОРІВНЯННЯ ЯКОСТІ МОДЕЛЕЙ")
print("=" * 70)
print(metrics.to_string(index=False))
print("=" * 70)


joblib.dump(cb, "models/catboost_model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("\nНавчання завершено")
print("CatBoost та LabelEncoder збережені для MVP")
