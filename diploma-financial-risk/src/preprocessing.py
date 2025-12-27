import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(df: pd.DataFrame):
    df = df.copy()
    le = LabelEncoder()
    df["service_category_encoded"] = le.fit_transform(df["service_category"])

    X = df[[
        "service_category_encoded",
        "avg_cost",
        "reimbursement_rate",
        "volume_per_month",
        "cost_variance",
        "payment_delay",
        "complications_rate"
    ]]

    y = df["risk_category"]

    return X, y, le
