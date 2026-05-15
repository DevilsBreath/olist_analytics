from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api import deps
from api.routers import metrics, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    deps.load_all()
    yield


app = FastAPI(
    title       = "E-Commerce Analytics API",
    description = "KPI metrics, churn prediction, low-review prediction with SHAP explanations.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

app.include_router(metrics.router)
app.include_router(predict.router)


@app.get("/ui", include_in_schema=False)
def dashboard():
    return FileResponse(Path("dashboard") / "index.html")


@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health_detail():
    return {
        "status"   : "ok",
        "models"   : ["XGBoost Churn", "LightGBM Low-Review"],
        "endpoints": [
            "GET  /metrics",
            "GET  /metrics/revenue",
            "GET  /metrics/delivery",
            "GET  /metrics/sellers",
            "GET  /cohort",
            "POST /predict/churn",
            "POST /predict/review",
        ],
    }
