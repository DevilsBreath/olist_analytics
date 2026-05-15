# Olist E-Commerce Analytics Platform

End-to-end data science project on the [Brazilian E-Commerce (Olist) dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).  
Covers the full DS stack: SQL analytics → feature engineering → ML models → SHAP explainability → FastAPI → interactive dashboard.

Built as a portfolio project targeting DS roles (Flipkart, Meesho, Fractal, Sigmoid).

---

## Architecture

```
Olist Dataset (8 CSVs, 100k+ orders)
        │  SQLAlchemy + psycopg2
        ▼
PostgreSQL — olist_db
  ├── 8 tables  (orders, customers, order_items, payments,
  │              reviews, products, sellers, translations)
  └── 4 SQL views (window functions, CTEs, cohort analysis)
        │  Python pipeline (validation + caching)
        ▼
pipeline_output/  ← CSV cache of all views
        │
        ▼
ML Layer
  ├── XGBoost  Churn Predictor    (SMOTE · PR-AUC 0.89 · ROC-AUC 0.68)
  ├── LightGBM Low-Review Predictor (SMOTE · PR-AUC 0.40 · ROC-AUC 0.69)
  └── SHAP    global + local explanations  →  shap_output/
        │
        ▼
FastAPI  (api/)
  ├── GET  /metrics  /metrics/revenue  /metrics/delivery  /metrics/sellers
  ├── GET  /cohort
  └── POST /predict/churn   /predict/review   (inline SHAP drivers)
        │
        ▼
Dashboard  (dashboard/index.html  →  http://localhost:8000/ui)
  Bootstrap 5 + Plotly.js  — Analytics · Churn Predictor · Review Risk
```

---

## Dataset

| Table | Rows | Description |
|---|---|---|
| orders | 99,441 | Order lifecycle + timestamps |
| customers | 99,441 | Customer + state |
| order_items | 112,650 | Items, prices, sellers |
| payments | 103,886 | Payment type, installments |
| reviews | 100,000 | Review score + comments |
| products | 32,951 | Category, dimensions |
| sellers | 3,095 | Seller location |
| product_category_translation | 71 | EN category names |

Download: [Kaggle — Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)  
Place all CSVs in `Data/` before running `01_db_setup.ipynb`.

---

## Project Structure

```
.
├── 01_db_setup.ipynb            # PostgreSQL schema creation + CSV ingestion
├── 02_data_pipeline.ipynb       # SQLAlchemy pull → validation → pipeline_output/
├── 03_ml_churn.ipynb            # Feature engineering + XGBoost + LightGBM
├── 04_shap_explainability.ipynb # SHAP global + local explanations
│
├── sql_query_used               # 4 SQL views (DDL)
├── pipeline_output/             # Cached CSVs from SQL views
├── models/                      # Saved model artifacts
│   ├── churn_xgb.json           # XGBoost churn model
│   ├── churn_encoders.pkl       # LabelEncoders for categorical features
│   ├── churn_meta.json          # Feature list + eval metrics
│   ├── review_lgb.txt           # LightGBM booster
│   ├── review_encoders.pkl
│   └── review_meta.json
├── shap_output/                 # SHAP plots (PNG) + driver JSONs
│
├── api/                         # FastAPI backend
│   ├── main.py                  # App + CORS + /ui route
│   ├── deps.py                  # Singleton model/data loader
│   ├── schemas.py               # Pydantic request/response models
│   └── routers/
│       ├── metrics.py           # Analytics endpoints
│       └── predict.py           # Prediction + SHAP endpoints
│
├── dashboard/
│   └── index.html               # Single-page dashboard (Bootstrap + Plotly)
│
└── requirements_api.txt         # API dependencies
```

---

## Quick Start (models pre-trained)

Requires Python 3.10+, PostgreSQL running with `olist_db` populated (see Full Setup below).

```bash
# 1. Clone
git clone https://github.com/DevilsBreath/olist_analytics
cd E-commerce_project

# 2. Install API dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 4. Open dashboard
# http://localhost:8000/ui

# 5. Interactive API docs
# http://localhost:8000/docs
```

---

## Full Reproduction (from raw CSVs)

### Prerequisites

- PostgreSQL 14+
- Conda or venv with Python 3.10
- Kaggle account (to download dataset)

### Step 1 — Database

```sql
-- In psql:
CREATE USER olist_user WITH PASSWORD '1234';
CREATE DATABASE olist_db OWNER olist_user;
GRANT ALL PRIVILEGES ON DATABASE olist_db TO olist_user;
```

### Step 2 — Environment

```bash
conda create -n olist 
conda activate olist
pip install -r requirements.txt
```

### Step 3 — Run notebooks in order

| Notebook | What it does | Output |
|---|---|---|
| `01_db_setup.ipynb` | Load 8 CSVs → PostgreSQL + create 4 SQL views | `olist_db` tables + views |
| `02_data_pipeline.ipynb` | Pull views → validate → cache | `pipeline_output/` |
| `03_ml_churn.ipynb` | Feature engineering, SMOTE, XGBoost + LightGBM | `models/` |
| `04_shap_explainability.ipynb` | SHAP global + local + save drivers | `shap_output/` |

---

## SQL Analytics Layer

Four PostgreSQL views built with advanced SQL:

| View | Technique | Business Question |
|---|---|---|
| `monthly_revenue` | `LAG()` window function | Revenue trend + MoM growth |
| `cohort_retention` | CTE + date truncation + self-join | What % of customers return each month? |
| `seller_ranking` | `DENSE_RANK()` partitioned by revenue | Who are the top sellers? |
| `delivery_delay_by_state` | Conditional aggregation | Which states have the worst delivery? |

---

## ML Models

### Model 1 — Customer Churn Predictor (XGBoost)

Churn definition: no purchase in 90 days after last order.

| Feature | Description |
|---|---|
| `frequency` | Number of distinct orders |
| `monetary` | Total spend (R$) |
| `avg_order_value` | Mean spend per order |
| `avg_review_score` | Mean review score (1–5) |
| `avg_delivery_delay` | Mean days early/late |
| `pct_late_orders` | Fraction of orders delivered late |
| `avg_installments` | Mean payment installments |
| `tenure_days` | Days between first and last order |
| `customer_state` | Brazilian state (encoded) |
| `top_category` | Most purchased category (encoded) |
| `top_payment_type` | Most used payment method (encoded) |

SMOTE applied to address 80/20 class imbalance.

| Metric | Score |
|---|---|
| PR-AUC | **0.89** |
| ROC-AUC | **0.68** |

### Model 2 — Low-Review Predictor (LightGBM)

Predicts whether an order will receive a 1–2 star review.

| Feature | Description |
|---|---|
| `price` | Item price (R$) |
| `freight_value` | Delivery cost (R$) |
| `delivery_delay` | Days late (positive = late) |
| `payment_installments` | Number of installments |
| `payment_type` | Payment method (encoded) |
| `category` | Product category (encoded) |

| Metric | Score |
|---|---|
| PR-AUC | **0.40** |
| ROC-AUC | **0.69** |

---

## API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

### Analytics Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/metrics` | KPI summary (revenue, orders, avg review, delivery delay, MoM growth) |
| GET | `/metrics/revenue?limit=24` | Monthly revenue time series with MoM growth % |
| GET | `/metrics/delivery?sort_by=avg_delay_days` | Delivery delay by Brazilian state |
| GET | `/metrics/sellers?top_n=20` | Top sellers by revenue + review score |
| GET | `/cohort?max_months=12` | Cohort retention matrix (pivot table) |

### Prediction Endpoints

**POST `/predict/churn`**

```json
// Request
{
  "frequency": 1,
  "monetary": 120.0,
  "avg_order_value": 120.0,
  "avg_review_score": 2.5,
  "avg_delivery_delay": 4.5,
  "pct_late_orders": 1.0,
  "avg_installments": 1.0,
  "tenure_days": 0,
  "customer_state": "SP",
  "top_category": "health_beauty",
  "top_payment_type": "boleto"
}

// Response
{
  "churn_probability": 0.7842,
  "churn_prediction": true,
  "threshold": 0.5,
  "base_value": -0.0081,
  "shap_drivers": [
    {"feature": "avg_review_score", "shap_value": 0.8993, "direction": "increases_risk"},
    {"feature": "avg_installments", "shap_value": 0.6643, "direction": "increases_risk"},
    ...
  ]
}
```

**POST `/predict/review`**

```json
// Request
{
  "price": 200.0,
  "freight_value": 30.0,
  "delivery_delay": 10.5,
  "payment_installments": 1,
  "payment_type": "boleto",
  "category": "electronics"
}

// Response
{
  "low_review_probability": 0.9289,
  "low_review_prediction": true,
  "threshold": 0.5,
  "base_value": 0.3639,
  "shap_drivers": [
    {"feature": "delivery_delay", "shap_value": 2.5406, "direction": "increases_risk"},
    ...
  ]
}
```

---

## Dashboard

Served at `http://localhost:8000/ui` (Bootstrap 5 + Plotly.js, no build step).

| Tab | Charts |
|---|---|
| Analytics | 6 KPI cards · Revenue trend + MoM growth (dual-axis) · Delivery delay by state · Cohort retention heatmap · Model metric badges |
| Churn Predictor | 11-field form → churn probability + HIGH/LOW badge + SHAP horizontal bar |
| Review Risk | 6-field form → low-review probability + SHAP horizontal bar |

SHAP bars are colored by direction: red = increases risk, green = decreases risk.

---

## SHAP Explainability

Global and local explanations for both models saved in `shap_output/`:

| File | Description |
|---|---|
| `churn_shap_beeswarm.png` | Global feature impact distribution |
| `churn_shap_bar.png` | Mean absolute SHAP by feature |
| `churn_waterfall_churned.png` | Why a high-risk customer is predicted to churn |
| `churn_waterfall_retained.png` | Why a retained customer stays |
| `churn_dep_delivery_delay.png` | Delivery delay vs SHAP (colored by review score) |
| `churn_force_top5.html` | Interactive force plot for top 5 churned customers |
| `churn_shap_drivers.json` | Top-3 SHAP drivers per test sample (used by API) |

Key finding: `avg_delivery_delay` is the strongest driver in both models — late deliveries directly predict both churn and low reviews.

---

## Key Findings

- **R$ 15.4M** total revenue across 96,476 delivered orders
- **Median MoM revenue growth: +7.1%** (peak month: Nov 2017)
- **80% churn rate** — most customers are single-purchase
- **12.8% low-review rate** — delivery delay is the #1 predictor
- Average delivery is **12.6 days early** vs estimate — but late orders correlate strongly with dissatisfaction
- Boleto (unbanked payment) users show higher churn vs credit card

---

## Tech Stack

| Layer | Tools |
|---|---|
| Database | PostgreSQL 14, psycopg2, SQLAlchemy |
| Data | pandas, numpy |
| ML | XGBoost, LightGBM, scikit-learn, imbalanced-learn (SMOTE) |
| Explainability | SHAP (TreeExplainer) |
| API | FastAPI, Pydantic v2, uvicorn |
| Frontend | Bootstrap 5, Plotly.js |
| Environment | Python 3.10, Conda |
