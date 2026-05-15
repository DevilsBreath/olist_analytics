from typing import Optional
from pydantic import BaseModel, Field


# ── Prediction request schemas ───────────────────────────────────────────────

class ChurnRequest(BaseModel):
    frequency: int             = Field(..., ge=1,       example=2,              description="Number of distinct orders")
    monetary: float            = Field(..., gt=0,       example=350.5,          description="Total spend (BRL)")
    avg_order_value: float     = Field(..., gt=0,       example=175.25,         description="Mean spend per order")
    avg_review_score: float    = Field(..., ge=1, le=5, example=4.0,            description="Mean review score (1–5)")
    avg_delivery_delay: float  = Field(...,             example=-5.2,           description="Mean delivery delay in days (negative = early)")
    pct_late_orders: float     = Field(..., ge=0, le=1, example=0.1,            description="Fraction of orders delivered late")
    avg_installments: float    = Field(..., ge=1,       example=3.0,            description="Mean payment installments")
    tenure_days: int           = Field(..., ge=0,       example=180,            description="Days between first and last order")
    customer_state: str        = Field(...,             example="SP",           description="Brazilian state code (e.g. SP, RJ)")
    top_category: str          = Field(...,             example="health_beauty", description="Most purchased product category")
    top_payment_type: str      = Field(...,             example="credit_card",  description="Most used payment type")
    recency: Optional[int]     = Field(None, ge=0,      example=45,             description="Days since last order (required if model trained with recency)")

    model_config = {"json_schema_extra": {"example": {
        "frequency": 2, "monetary": 350.5, "avg_order_value": 175.25,
        "avg_review_score": 3.5, "avg_delivery_delay": 2.1, "pct_late_orders": 0.5,
        "avg_installments": 4.0, "tenure_days": 30, "customer_state": "SP",
        "top_category": "health_beauty", "top_payment_type": "credit_card",
    }}}


class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    threshold: float
    base_value: float
    shap_drivers: list[dict]


class ReviewRequest(BaseModel):
    price: float               = Field(..., gt=0,  example=150.0,        description="Item price (BRL)")
    freight_value: float       = Field(..., ge=0,  example=20.5,         description="Freight cost (BRL)")
    delivery_delay: float      = Field(...,        example=3.5,          description="Actual minus estimated delivery (days)")
    payment_installments: float = Field(..., ge=1, example=3.0,          description="Number of payment installments")
    payment_type: str          = Field(...,        example="credit_card", description="Payment method")
    category: str              = Field(...,        example="electronics", description="Product category")

    model_config = {"json_schema_extra": {"example": {
        "price": 150.0, "freight_value": 20.5, "delivery_delay": 5.0,
        "payment_installments": 3.0, "payment_type": "credit_card", "category": "electronics",
    }}}


class ReviewResponse(BaseModel):
    low_review_probability: float
    low_review_prediction: bool
    threshold: float
    base_value: float
    shap_drivers: list[dict]


# ── Metrics response schemas ─────────────────────────────────────────────────

class KPISummary(BaseModel):
    total_revenue: float
    total_orders: int
    avg_review_score: float
    avg_delivery_delay_days: float
    best_revenue_month: str
    avg_mom_growth_pct: float
    model_metrics: dict
