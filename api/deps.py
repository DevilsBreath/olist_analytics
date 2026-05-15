"""
Singleton state: models + pipeline CSVs loaded once at startup via lifespan.
Access any artifact with deps.get("key").
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
import lightgbm as lgb

MODELS_DIR   = Path("models")
PIPELINE_DIR = Path("pipeline_output")

_state: dict = {}


def load_all() -> None:
    # ── Churn model (XGBoost) ────────────────────────────────────────────────
    xgb_model = XGBClassifier()
    xgb_model.load_model(MODELS_DIR / "churn_xgb.json")

    with open(MODELS_DIR / "churn_meta.json") as f:
        churn_meta = json.load(f)
    with open(MODELS_DIR / "churn_encoders.pkl", "rb") as f:
        churn_enc = pickle.load(f)

    churn_explainer = shap.TreeExplainer(xgb_model)

    # ── Review model (LightGBM Booster) ─────────────────────────────────────
    lgb_booster = lgb.Booster(model_file=str(MODELS_DIR / "review_lgb.txt"))

    with open(MODELS_DIR / "review_meta.json") as f:
        review_meta = json.load(f)
    with open(MODELS_DIR / "review_encoders.pkl", "rb") as f:
        review_enc = pickle.load(f)

    review_explainer = shap.TreeExplainer(lgb_booster)

    # ── Pipeline CSV cache ───────────────────────────────────────────────────
    monthly_revenue  = pd.read_csv(PIPELINE_DIR / "monthly_revenue.csv")
    cohort_retention = pd.read_csv(PIPELINE_DIR / "cohort_retention.csv")
    seller_ranking   = pd.read_csv(PIPELINE_DIR / "seller_ranking.csv")
    delivery_delay   = pd.read_csv(PIPELINE_DIR / "delivery_delay_by_state.csv")

    _state.update(
        xgb_model        = xgb_model,
        churn_meta       = churn_meta,
        churn_enc        = churn_enc,
        churn_explainer  = churn_explainer,
        lgb_booster      = lgb_booster,
        review_meta      = review_meta,
        review_enc       = review_enc,
        review_explainer = review_explainer,
        monthly_revenue  = monthly_revenue,
        cohort_retention = cohort_retention,
        seller_ranking   = seller_ranking,
        delivery_delay   = delivery_delay,
    )
    print("All artifacts loaded.")


def get(key: str):
    return _state[key]
