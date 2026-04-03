from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import uuid
import io
import os
import pickle

engine = create_engine("postgresql://fraud_user:password123@localhost:5432/fraud_db")
app = FastAPI(title="Fraud Detection API")

demo_transactions = []

REQUIRED_UPLOAD_COLUMNS = {
    "transaction_id", "customer_id", "merchant", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "timestamp", "merchant_category", "payment_method", "city",
    "device_type", "hour_of_day", "is_international"
}

FRAUD_TYPE_MAP = {
    "CASH_OUT": 0,
    "DEBIT": 1,
    "PAYMENT": 2,
    "TRANSFER": 3
}

FRAUD_THRESHOLD = 0.50
HIGH_RISK_THRESHOLD = 0.80

rf_model = None
xgb_model = None
category_model = None
category_feature_columns = None
category_label_encoder = None


def load_pickle_model(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


try:
    rf_model = load_pickle_model("rf_model.pkl")
except Exception:
    rf_model = None

try:
    xgb_model = load_pickle_model("xgb_model.pkl")
except Exception:
    xgb_model = None

try:
    category_model = load_pickle_model("category_model.pkl")
except Exception:
    category_model = None

try:
    category_feature_columns = load_pickle_model("category_feature_columns.pkl")
except Exception:
    category_feature_columns = None

try:
    category_label_encoder = load_pickle_model("category_label_encoder.pkl")
except Exception:
    category_label_encoder = None


class TransactionInput(BaseModel):
    customer_id: str
    merchant: str
    type: str
    amount: float = Field(..., ge=0)
    merchant_category: str = "Other"
    payment_method: str = "Card"
    city: str = "Calgary"
    device_type: str = "Mobile"
    hour_of_day: int = 12
    is_international: int = 0


def get_customer_latest_context(customer_id: str):
    customer_rows = [r for r in demo_transactions if r.get("customer_id") == customer_id]
    if not customer_rows:
        return {
            "oldbalanceOrg": 5000.0,
            "newbalanceOrig": 5000.0,
            "oldbalanceDest": 1000.0,
            "newbalanceDest": 1000.0,
            "step": 1,
        }

    customer_rows = sorted(customer_rows, key=lambda x: x.get("processed_at", ""), reverse=True)
    latest = customer_rows[0]

    return {
        "oldbalanceOrg": float(latest.get("newbalanceOrig", latest.get("oldbalanceOrg", 5000.0))),
        "newbalanceOrig": float(latest.get("newbalanceOrig", latest.get("oldbalanceOrg", 5000.0))),
        "oldbalanceDest": float(latest.get("newbalanceDest", latest.get("oldbalanceDest", 1000.0))),
        "newbalanceDest": float(latest.get("newbalanceDest", latest.get("oldbalanceDest", 1000.0))),
        "step": int(latest.get("step", 1)) + 1 if str(latest.get("step", "")).isdigit() else 1,
    }


def fraud_fallback_score(record: dict):
    fraud_probability = 0.03
    tx_type = str(record.get("type", "")).upper()
    amount = float(record.get("amount", 0))
    oldbalanceOrg = float(record.get("oldbalanceOrg", 0))
    newbalanceOrig = float(record.get("newbalanceOrig", 0))
    oldbalanceDest = float(record.get("oldbalanceDest", 0))
    newbalanceDest = float(record.get("newbalanceDest", 0))

    if tx_type in ["TRANSFER", "CASH_OUT"]:
        fraud_probability += 0.30
    if amount >= 3000:
        fraud_probability += 0.15
    if amount >= 7000:
        fraud_probability += 0.20
    if oldbalanceOrg > 0 and newbalanceOrig == 0:
        fraud_probability += 0.20
    if oldbalanceOrg > 0 and amount >= 0.8 * oldbalanceOrg:
        fraud_probability += 0.15
    if oldbalanceDest == 0 and newbalanceDest > 0:
        fraud_probability += 0.10
    if newbalanceDest > oldbalanceDest and amount > 2000:
        fraud_probability += 0.10

    return round(min(max(fraud_probability, 0.01), 0.99), 4)


def prepare_fraud_features(record: dict):
    row = {
        "step": float(record.get("step", 1)),
        "type": FRAUD_TYPE_MAP.get(str(record.get("type", "")).upper(), 0),
        "amount": float(record.get("amount", 0)),
        "oldbalanceOrg": float(record.get("oldbalanceOrg", 0)),
        "newbalanceOrig": float(record.get("newbalanceOrig", 0)),
        "oldbalanceDest": float(record.get("oldbalanceDest", 0)),
        "newbalanceDest": float(record.get("newbalanceDest", 0)),
    }
    return pd.DataFrame([row])


def prepare_category_features(record: dict):
    df = pd.DataFrame([{
        "merchant": record.get("merchant", "Unknown"),
        "type": record.get("type", "Unknown"),
        "amount": float(record.get("amount", 0)),
        "oldbalanceOrg": float(record.get("oldbalanceOrg", 0)),
        "newbalanceOrig": float(record.get("newbalanceOrig", 0)),
        "oldbalanceDest": float(record.get("oldbalanceDest", 0)),
        "newbalanceDest": float(record.get("newbalanceDest", 0)),
        "merchant_category": record.get("merchant_category", "Unknown"),
        "payment_method": record.get("payment_method", "Unknown"),
        "city": record.get("city", "Unknown"),
        "device_type": record.get("device_type", "Unknown"),
        "hour_of_day": int(record.get("hour_of_day", 12)),
        "is_international": int(record.get("is_international", 0)),
    }])

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(0)

    df = pd.get_dummies(df, drop_first=True)

    if category_feature_columns is not None:
        for col in category_feature_columns:
            if col not in df.columns:
                df[col] = 0
        extra_cols = [c for c in df.columns if c not in category_feature_columns]
        if extra_cols:
            df = df.drop(columns=extra_cols, errors="ignore")
        df = df[category_feature_columns]

    return df


def fallback_spending_category(record: dict):
    merchant = str(record.get("merchant", "")).lower()
    merchant_hint = str(record.get("merchant_category", "")).lower()
    tx_type = str(record.get("type", "")).upper()
    amount = float(record.get("amount", 0))

    if any(x in merchant for x in ["royalbank", "globalbank", "credit union", "loan", "invest"]):
        return "Finance"
    if any(x in merchant for x in ["university", "college", "school", "udemy", "coursera"]):
        return "Education"
    if any(x in merchant for x in ["flix", "netflix", "spotify", "cinema", "gamestream"]):
        return "Entertainment"
    if any(x in merchant for x in ["tel", "datacom", "mail", "sendfast", "phone", "internet"]):
        return "Communication Services"
    if any(x in merchant for x in ["care", "pharma", "clinic", "health", "community"]):
        return "Health and Community Services"
    if any(x in merchant for x in ["biz", "office", "lease", "property", "desk"]):
        return "Property and Business Services"
    if any(x in merchant for x in ["amazon", "mall", "market", "store", "plaza", "retail", "apple"]):
        return "Retail Trade"
    if any(x in merchant for x in ["fuel", "ride", "cabs", "transport", "transway"]):
        return "Services to Transport"
    if any(x in merchant for x in ["tour", "flight", "travel", "journey", "globe"]):
        return "Travel and Trade"
    if any(x in merchant for x in ["consult", "law", "sage", "personal", "legal"]):
        return "Professional and Personal Services"

    if "bank" in merchant_hint or "investment" in merchant_hint:
        return "Finance"
    if "tuition" in merchant_hint or "education" in merchant_hint:
        return "Education"
    if "stream" in merchant_hint or "media" in merchant_hint:
        return "Entertainment"
    if "telecom" in merchant_hint or "internet" in merchant_hint:
        return "Communication Services"
    if "clinic" in merchant_hint or "pharmacy" in merchant_hint:
        return "Health and Community Services"
    if "office" in merchant_hint or "business" in merchant_hint or "property" in merchant_hint:
        return "Property and Business Services"
    if "retail" in merchant_hint or "shopping" in merchant_hint:
        return "Retail Trade"
    if "transport" in merchant_hint or "fuel" in merchant_hint:
        return "Services to Transport"
    if "travel" in merchant_hint or "airline" in merchant_hint:
        return "Travel and Trade"
    if "consulting" in merchant_hint or "legal" in merchant_hint or "personal" in merchant_hint:
        return "Professional and Personal Services"

    if tx_type in ["TRANSFER", "CASH_OUT"] and amount >= 3000:
        return "Finance"

    return "Retail Trade"


def predict_spending_category_model(record: dict):
    if category_model is None:
        return None

    try:
        X = prepare_category_features(record)
        pred = category_model.predict(X)[0]

        if category_label_encoder is not None:
            pred = category_label_encoder.inverse_transform([int(pred)])[0]

        return str(pred)
    except Exception:
        return None


def predict_spending_category(record: dict):
    model_prediction = predict_spending_category_model(record)
    fallback_prediction = fallback_spending_category(record)

    suspicious_single_class = {
        "Trade, Professional and Personal Services",
        "Professional and Personal Services"
    }

    if model_prediction is None or model_prediction in suspicious_single_class:
        return fallback_prediction

    return model_prediction


def score_transaction_record(record: dict):
    rf_prob = None
    xgb_prob = None

    try:
        X_fraud = prepare_fraud_features(record)
        if rf_model is not None:
            rf_prob = float(rf_model.predict_proba(X_fraud)[0][1])
        if xgb_model is not None:
            xgb_prob = float(xgb_model.predict_proba(X_fraud)[0][1])
    except Exception:
        rf_prob = None
        xgb_prob = None

    fallback_prob = fraud_fallback_score(record)
    final_prob = xgb_prob if xgb_prob is not None else (rf_prob if rf_prob is not None else fallback_prob)
    final_label = "Fraud" if final_prob >= FRAUD_THRESHOLD else "Not Fraud"
    predicted_category = predict_spending_category(record)

    record["rf_probability"] = round(rf_prob, 4) if rf_prob is not None else None
    record["xgb_probability"] = round(xgb_prob, 4) if xgb_prob is not None else None
    record["fraud_probability"] = round(final_prob, 4)
    record["fraud_label"] = final_label
    record["risk_level"] = (
        "High Risk" if final_prob >= HIGH_RISK_THRESHOLD
        else "Medium Risk" if final_prob >= FRAUD_THRESHOLD
        else "Low Risk"
    )
    record["predicted_category"] = predicted_category
    record["ingestion_status"] = "processed"
    record["processed_at"] = datetime.utcnow().isoformat()
    return record


@app.get("/")
def home():
    return {"message": "API is running"}


@app.get("/health")
def health():
    db_ok = True
    try:
        pd.read_sql("SELECT 1", engine)
    except Exception:
        db_ok = False

    return {
        "status": "ok" if db_ok else "degraded",
        "database": db_ok,
        "rf_model_loaded": rf_model is not None,
        "xgb_model_loaded": xgb_model is not None,
        "category_model_loaded": category_model is not None,
        "category_feature_columns_loaded": category_feature_columns is not None,
        "category_label_encoder_loaded": category_label_encoder is not None
    }


@app.post("/transactions/upload")
async def upload_transactions(file: UploadFile = File(...)):
    global demo_transactions

    filename = (file.filename or "").lower()
    if not (filename.endswith(".csv") or filename.endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="Only CSV or XLSX files are allowed")

    content = await file.read()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

    missing = REQUIRED_UPLOAD_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {sorted(missing)}")

    processed_records = []
    existing = demo_transactions.copy()

    for i, row in df.iterrows():
        record = row.to_dict()
        record["step"] = len(existing) + i + 1
        if pd.isna(record.get("transaction_id")) or str(record.get("transaction_id")).strip() == "":
            record["transaction_id"] = str(uuid.uuid4())
        scored = score_transaction_record(record)
        processed_records.append(scored)

    demo_transactions.extend(processed_records)
    demo_transactions.sort(key=lambda x: x.get("processed_at", x.get("timestamp", "")), reverse=True)

    if len(demo_transactions) > 1000:
        demo_transactions = demo_transactions[:1000]

    return {
        "status": "success",
        "rows_loaded": len(processed_records),
        "total_transactions_after_upload": len(demo_transactions),
        "message": "File uploaded and appended successfully"
    }


@app.get("/customers")
def get_customers():
    return sorted(list({r.get("customer_id") for r in demo_transactions if r.get("customer_id")}))


@app.get("/merchants")
def get_merchants():
    return sorted(list({r.get("merchant") for r in demo_transactions if r.get("merchant")}))


@app.get("/customer-context/{customer_id}")
def customer_context(customer_id: str):
    return get_customer_latest_context(customer_id)


@app.post("/transactions/ingest")
def ingest_transaction(tx: TransactionInput):
    context = get_customer_latest_context(tx.customer_id)

    oldbalanceOrg = float(context["oldbalanceOrg"])
    newbalanceOrig = max(oldbalanceOrg - float(tx.amount), 0.0)
    oldbalanceDest = float(context["oldbalanceDest"])
    newbalanceDest = oldbalanceDest + float(tx.amount)
    step = int(context["step"])

    record = {
        "transaction_id": str(uuid.uuid4()),
        "customer_id": tx.customer_id,
        "merchant": tx.merchant,
        "type": tx.type.upper(),
        "amount": float(tx.amount),
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "merchant_category": tx.merchant_category,
        "payment_method": tx.payment_method,
        "city": tx.city,
        "device_type": tx.device_type,
        "hour_of_day": tx.hour_of_day,
        "is_international": tx.is_international,
        "step": step,
    }

    scored = score_transaction_record(record)
    demo_transactions.insert(0, scored)

    if len(demo_transactions) > 1000:
        demo_transactions.pop()

    return scored


@app.get("/transactions/recent")
def get_recent_transactions():
    return demo_transactions[:150]


@app.get("/metrics")
def get_metrics():
    if not demo_transactions:
        return {
            "total_transactions": 0,
            "fraud_count": 0,
            "fraud_rate": 0.0,
            "flagged_amount": 0.0,
            "high_risk_transactions": 0
        }

    df = pd.DataFrame(demo_transactions)
    total_transactions = len(df)
    fraud_count = int((df["fraud_label"] == "Fraud").sum())
    fraud_rate = fraud_count / total_transactions if total_transactions > 0 else 0.0
    flagged_amount = float(df.loc[df["fraud_label"] == "Fraud", "amount"].sum())
    high_risk_transactions = int((df["fraud_probability"] >= HIGH_RISK_THRESHOLD).sum())

    return {
        "total_transactions": total_transactions,
        "fraud_count": fraud_count,
        "fraud_rate": round(fraud_rate, 4),
        "flagged_amount": round(flagged_amount, 2),
        "high_risk_transactions": high_risk_transactions
    }


@app.get("/spending/summary")
def get_spending_summary():
    if not demo_transactions:
        return []

    df = pd.DataFrame(demo_transactions)
    summary = (
        df.groupby("predicted_category", as_index=False)
        .agg(
            transaction_count=("transaction_id", "count"),
            total_amount=("amount", "sum"),
            fraud_count=("fraud_label", lambda s: int((s == "Fraud").sum()))
        )
        .sort_values("total_amount", ascending=False)
        .rename(columns={"predicted_category": "category"})
    )
    summary["fraud_rate"] = summary["fraud_count"] / summary["transaction_count"]
    return summary.to_dict(orient="records")


@app.get("/spending/merchant-treemap")
def spending_merchant_treemap():
    if not demo_transactions:
        return []

    df = pd.DataFrame(demo_transactions)
    grouped = (
        df.groupby(["predicted_category", "merchant"], as_index=False)
        .agg(total_amount=("amount", "sum"))
    )
    return grouped.to_dict(orient="records")


@app.get("/fraud/hourly-bars")
def fraud_hourly_bars():
    if not demo_transactions:
        return []

    df = pd.DataFrame(demo_transactions)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    grouped = (
        df.groupby("hour", as_index=False)
        .agg(
            fraud_count=("fraud_label", lambda s: int((s == "Fraud").sum())),
            total_transactions=("transaction_id", "count"),
            flagged_amount=("amount", lambda s: float(df.loc[s.index][df.loc[s.index, "fraud_label"] == "Fraud"]["amount"].sum()))
        )
        .sort_values("hour")
    )

    return grouped.to_dict(orient="records")


@app.get("/fraud/by-type")
def fraud_by_type():
    if not demo_transactions:
        return []

    df = pd.DataFrame(demo_transactions)
    grouped = (
        df.groupby("type", as_index=False)
        .agg(
            transaction_count=("transaction_id", "count"),
            fraud_count=("fraud_label", lambda s: int((s == "Fraud").sum()))
        )
    )
    grouped["fraud_rate"] = grouped["fraud_count"] / grouped["transaction_count"]
    return grouped.to_dict(orient="records")


@app.get("/fraud/top-merchants")
def fraud_top_merchants():
    if not demo_transactions:
        return []

    df = pd.DataFrame(demo_transactions)
    fraud_df = df[df["fraud_label"] == "Fraud"]

    if fraud_df.empty:
        return []

    grouped = (
        fraud_df.groupby("merchant", as_index=False)
        .agg(
            fraud_count=("transaction_id", "count"),
            flagged_amount=("amount", "sum")
        )
        .sort_values(["flagged_amount", "fraud_count"], ascending=False)
        .head(10)
    )
    return grouped.to_dict(orient="records")