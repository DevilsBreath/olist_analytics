import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from api import deps
from api.schemas import ChurnRequest, ChurnResponse, ReviewRequest, ReviewResponse

router = APIRouter(prefix="/predict", tags=["Predictions"])

CHURN_THRESHOLD  = 0.5
REVIEW_THRESHOLD = 0.5


def _encode_row(row: dict, encoders: dict, cat_cols: list) -> dict:
    """Apply saved LabelEncoders. Unknown category → class index 0 (most common fallback)."""
    out = dict(row)
    for col in cat_cols:
        le  = encoders[col]
        val = str(out[col])
        out[col] = int(le.transform([val])[0]) if val in le.classes_ else 0
    return out


def _shap_drivers(shap_values: np.ndarray, feature_names: list, top_n: int = 5) -> list[dict]:
    """Top-n features by |SHAP|, with direction label."""
    idx = np.argsort(np.abs(shap_values))[::-1][:top_n]
    return [
        {
            "feature"   : feature_names[i],
            "shap_value": round(float(shap_values[i]), 4),
            "direction" : "increases_risk" if shap_values[i] > 0 else "decreases_risk",
        }
        for i in idx
    ]


@router.post("/churn", response_model=ChurnResponse)
def predict_churn(req: ChurnRequest):
    """
    Predict churn probability for a customer.
    Returns probability, binary prediction, and top-5 SHAP feature drivers.
    """
    xgb_model   = deps.get("xgb_model")
    churn_meta  = deps.get("churn_meta")
    churn_enc   = deps.get("churn_enc")
    explainer   = deps.get("churn_explainer")

    feature_cols = list(churn_meta["feature_cols"])
    cat_cols     = list(churn_meta["cat_cols"])

    # handle model trained with recency (12 features vs 11 in meta)
    if xgb_model.n_features_in_ == len(feature_cols) + 1:
        if req.recency is None:
            raise HTTPException(
                status_code=422,
                detail="This model was trained with 'recency'. Provide 'recency' (days since last order).",
            )
        feature_cols = ["recency"] + feature_cols

    row = _encode_row(req.model_dump(), churn_enc, cat_cols)
    X   = pd.DataFrame([{col: row[col] for col in feature_cols}])

    prob    = float(xgb_model.predict_proba(X)[0, 1])
    sv_obj  = explainer(X)
    sv      = sv_obj.values[0]
    base    = float(sv_obj.base_values[0])

    return ChurnResponse(
        churn_probability = round(prob, 4),
        churn_prediction  = prob >= CHURN_THRESHOLD,
        threshold         = CHURN_THRESHOLD,
        base_value        = round(base, 4),
        shap_drivers      = _shap_drivers(sv, list(X.columns)),
    )


@router.post("/review", response_model=ReviewResponse)
def predict_review(req: ReviewRequest):
    """
    Predict whether an order will receive a low review (score ≤ 2).
    Returns probability, binary prediction, and top-5 SHAP feature drivers.
    """
    lgb_booster  = deps.get("lgb_booster")
    review_meta  = deps.get("review_meta")
    review_enc   = deps.get("review_enc")
    explainer    = deps.get("review_explainer")

    feature_cols = list(review_meta["feature_cols"])
    cat_cols     = list(review_meta["cat_cols"])

    row = _encode_row(req.model_dump(), review_enc, cat_cols)
    X   = pd.DataFrame([{col: row[col] for col in feature_cols}])

    prob    = float(lgb_booster.predict(X)[0])
    sv_obj  = explainer(X)
    sv      = sv_obj.values[0]
    base    = float(sv_obj.base_values[0])

    return ReviewResponse(
        low_review_probability = round(prob, 4),
        low_review_prediction  = prob >= REVIEW_THRESHOLD,
        threshold              = REVIEW_THRESHOLD,
        base_value             = round(base, 4),
        shap_drivers           = _shap_drivers(sv, list(X.columns)),
    )
