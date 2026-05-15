import math
from typing import Annotated
from fastapi import APIRouter, Query
from api import deps
from api.schemas import KPISummary

router = APIRouter()


def _nan_to_none(records: list[dict]) -> list[dict]:
    """Replace float NaN with None so FastAPI serializes as JSON null."""
    return [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
        for row in records
    ]


@router.get("/metrics", response_model=KPISummary, tags=["Analytics"])
def get_metrics():
    """Aggregated KPIs from SQL pipeline views."""
    rev         = deps.get("monthly_revenue")
    delay       = deps.get("delivery_delay")
    sellers     = deps.get("seller_ranking")
    churn_meta  = deps.get("churn_meta")
    review_meta = deps.get("review_meta")

    total_revenue = float(rev["revenue"].sum())
    total_orders  = int(delay["total_orders"].sum())

    weighted_score = (sellers["avg_review_score"] * sellers["total_orders"]).sum()
    avg_review     = float(weighted_score / sellers["total_orders"].sum())

    avg_delay = float(delay["avg_delay_days"].mean())

    clean_growth = rev["mom_growth_pct"].replace([float("inf"), -float("inf")], float("nan")).dropna()
    avg_mom      = float(clean_growth.median())   # median; mean distorted by Dec-2016 partial-month spike

    best_month = str(rev.loc[rev["revenue"].idxmax(), "month"])

    return KPISummary(
        total_revenue           = round(total_revenue, 2),
        total_orders            = total_orders,
        avg_review_score        = round(avg_review, 3),
        avg_delivery_delay_days = round(avg_delay, 3),
        best_revenue_month      = best_month,
        avg_mom_growth_pct      = round(avg_mom, 2),
        model_metrics           = {
            "churn_model" : {"pr_auc": churn_meta["pr_auc"],  "roc_auc": churn_meta["roc_auc"]},
            "review_model": {"pr_auc": review_meta["pr_auc"], "roc_auc": review_meta["roc_auc"]},
        },
    )


@router.get("/metrics/revenue", tags=["Analytics"])
def get_revenue(
    limit: Annotated[int, Query(ge=1, le=100, description="Max months to return")] = 24,
):
    """Monthly revenue time series with MoM growth %."""
    df = deps.get("monthly_revenue").tail(limit)
    return _nan_to_none(df.to_dict(orient="records"))


@router.get("/metrics/delivery", tags=["Analytics"])
def get_delivery(
    sort_by: Annotated[str, Query(enum=["avg_delay_days", "total_orders"])] = "avg_delay_days",
):
    """Delivery delay by Brazilian state."""
    df = deps.get("delivery_delay").copy()
    df = df.sort_values(sort_by, ascending=False)
    return df.to_dict(orient="records")


@router.get("/metrics/sellers", tags=["Analytics"])
def get_sellers(
    top_n: Annotated[int, Query(ge=1, le=100)] = 20,
):
    """Top sellers by revenue with review score."""
    df = deps.get("seller_ranking").head(top_n)
    return df.to_dict(orient="records")


@router.get("/cohort", tags=["Analytics"])
def get_cohort(
    max_months: Annotated[int, Query(ge=1, le=24, description="Max months_since_first to include")] = 12,
):
    """
    Cohort retention matrix.
    Returns {cohort_months, matrix} where matrix rows are
    {cohort_month, "0": 1.0, "1": 0.xx, ...} — ready for heatmap.
    """
    df = deps.get("cohort_retention").copy()

    df["month_num"] = df["months_since_first"].round().astype(int)
    df = df[df["month_num"] <= max_months]

    pivot = (
        df.pivot_table(
            index="cohort_month",
            columns="month_num",
            values="retention_rate",
            aggfunc="mean",
        )
        .reset_index()
    )
    pivot.columns = [str(c) for c in pivot.columns]

    return {
        "cohort_months": pivot["cohort_month"].tolist(),
        "max_period"   : max_months,
        "matrix"       : _nan_to_none(pivot.to_dict(orient="records")),
    }
