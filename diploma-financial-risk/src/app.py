from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI(title="Financial Risk Assessment MVP")

templates = Jinja2Templates(directory="templates")

model = joblib.load("models/catboost_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

FEATURE_COLUMNS = [
    "service_category",
    "avg_cost",
    "reimbursement_rate",
    "volume_per_month",
    "cost_variance",
    "payment_delay",
    "complications_rate",
]

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

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "services": SERVICE_CATEGORIES,
            "result": None
        }
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()

    try:
        data = {
            "service_category": form["service_category"],
            "avg_cost": float(form["avg_cost"]),
            "reimbursement_rate": float(form["reimbursement_rate"]),
            "volume_per_month": float(form["volume_per_month"]),
            "cost_variance": float(form["cost_variance"]),
            "payment_delay": float(form["payment_delay"]),
            "complications_rate": float(form["complications_rate"]),
        }

        X = pd.DataFrame([data], columns=FEATURE_COLUMNS)

        pred = int(model.predict(X)[0])
        risk_label = label_encoder.inverse_transform([pred])[0]

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "services": SERVICE_CATEGORIES,
                "result": risk_label
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "services": SERVICE_CATEGORIES,
                "result": f"Помилка: {e}"
            }
        )
