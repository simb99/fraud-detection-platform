import os
import time
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Fraud Detection & Spending Intelligence",
    page_icon="🚨",
    layout="wide"
)

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DEFAULT_FRAUD_THRESHOLD = 0.50
HIGH_RISK_THRESHOLD = 0.80

st.title("🚨 Fraud Detection & Spending Categorization")
st.caption("Upload a starter file, score transactions with ML models, then investigate fraud and spending behavior")

if "last_result" not in st.session_state:
    st.session_state.last_result = None


@st.cache_data(ttl=20)
def get_json(endpoint):
    r = requests.get(f"{BASE_URL}{endpoint}", timeout=30)
    r.raise_for_status()
    return r.json()


def post_json(endpoint, payload):
    r = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def post_file(endpoint, uploaded_file):
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    r = requests.post(f"{BASE_URL}{endpoint}", files=files, timeout=120)
    r.raise_for_status()
    return r.json()


def risk_badge(prob):
    if prob >= HIGH_RISK_THRESHOLD:
        return "🟣 High Risk"
    elif prob >= DEFAULT_FRAUD_THRESHOLD:
        return "🔴 Fraud"
    return "🟢 Low Risk"


def style_transactions(df):
    def row_style(row):
        prob = float(row.get("fraud_probability", 0))
        label = str(row.get("fraud_label", ""))

        if prob >= HIGH_RISK_THRESHOLD:
            return ["background-color: #efe3ff; color: #5b21b6; font-weight: 700;"] * len(row)
        elif label == "Fraud":
            return ["background-color: #fde2e1; color: #7a1c1c; font-weight: 600;"] * len(row)
        elif prob >= DEFAULT_FRAUD_THRESHOLD:
            return ["background-color: #fff1d6; color: #7a4b00;"] * len(row)
        else:
            return ["background-color: #eef7ee; color: #1f5e2d;"] * len(row)

    return df.style.apply(row_style, axis=1)


st.sidebar.header("System")
refresh = st.sidebar.button("Refresh dashboard")
threshold = st.sidebar.slider("Fraud alert threshold", 0.0, 1.0, DEFAULT_FRAUD_THRESHOLD, 0.01)

if refresh:
    st.cache_data.clear()

try:
    health = get_json("/health")
    st.sidebar.success(f"API: {health.get('status', 'unknown')}")
    st.sidebar.write(f"RF loaded: {health.get('rf_model_loaded', False)}")
    st.sidebar.write(f"XGB loaded: {health.get('xgb_model_loaded', False)}")
    st.sidebar.write(f"Category model loaded: {health.get('category_model_loaded', False)}")
except Exception:
    st.sidebar.error("Backend unavailable")
    st.error(f"Could not connect to API at {BASE_URL}")
    st.stop()

try:
    metrics = get_json("/metrics")
except Exception:
    metrics = {
        "total_transactions": 0,
        "fraud_count": 0,
        "fraud_rate": 0.0,
        "flagged_amount": 0.0,
        "high_risk_transactions": 0
    }

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Transactions", f"{metrics.get('total_transactions', 0):,}")
k2.metric("Fraud Count", f"{metrics.get('fraud_count', 0):,}")
k3.metric("Fraud Rate", f"{metrics.get('fraud_rate', 0):.2%}")
k4.metric("Flagged Amount", f"${metrics.get('flagged_amount', 0):,.2f}")
k5.metric("High-Risk Transactions", f"{metrics.get('high_risk_transactions', 0):,}")

st.divider()

top_tab1, top_tab2 = st.tabs(["Upload Starter File", "Add Live Transaction"])

with top_tab1:
    st.subheader("Upload Starter File")
    uploaded_file = st.file_uploader("Choose CSV or XLSX file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                preview_df = pd.read_csv(uploaded_file)
            else:
                preview_df = pd.read_excel(uploaded_file)

            st.write("Preview")
            st.dataframe(preview_df.head(10), use_container_width=True)

            if st.button("Process Uploaded File"):
                uploaded_file.seek(0)
                result = post_file("/transactions/upload", uploaded_file)
                st.success(result.get("message", "Upload completed"))
                st.info(f"Rows loaded this upload: {result.get('rows_loaded', 0)}")
                st.info(f"Total transactions now: {result.get('total_transactions_after_upload', 0)}")
                st.cache_data.clear()
                st.rerun()
        except Exception as e:
            st.error(f"Could not read/process file: {e}")

with top_tab2:
    st.subheader("Add Live Transaction")

    try:
        customers = get_json("/customers")
        merchants = get_json("/merchants")
    except Exception:
        customers = []
        merchants = []

    if not customers:
        st.warning("Upload a starter file first.")
    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            customer_id = st.selectbox("Customer", customers)
            tx_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
            amount = st.number_input("Amount", min_value=0.0, value=250.0, step=10.0)

        with c2:
            merchant_choice = st.selectbox("Merchant", merchants + ["Other"])
            merchant = st.text_input("New Merchant", "NewMerchant") if merchant_choice == "Other" else merchant_choice

            merchant_category = st.selectbox(
                "Merchant Type Hint",
                [
                    "Banking Service",
                    "Tuition Provider",
                    "Streaming Platform",
                    "Telecom Provider",
                    "Clinic or Pharmacy",
                    "Office or Property Vendor",
                    "Retail Store",
                    "Ride or Fuel Service",
                    "Travel Agency or Airline",
                    "Consulting or Legal Service",
                    "Other"
                ],
                help="This is a broad merchant hint, not the final predicted spending category."
            )

            payment_method = st.selectbox("Payment Method", ["Card", "Bank", "Cash"])

        with c3:
            city = st.selectbox("City", ["Calgary", "Edmonton", "Toronto", "Vancouver"])
            device_type = st.selectbox("Device Type", ["Mobile", "Web", "ATM", "CardTerminal"])
            hour_of_day = st.slider("Hour of Day", 0, 23, 12)
            is_international = st.selectbox(
                "International?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        try:
            context = get_json(f"/customer-context/{customer_id}")
            st.caption(
                f"Customer context → Origin balance: ${context.get('oldbalanceOrg', 0):,.2f}, "
                f"Destination balance: ${context.get('oldbalanceDest', 0):,.2f}"
            )
        except Exception:
            pass

        st.caption(
            "Note: the app uses merchant details and transaction behavior to predict a broader spending category."
        )

        if st.button("Send Live Transaction"):
            payload = {
                "customer_id": customer_id,
                "merchant": merchant,
                "type": tx_type,
                "amount": float(amount),
                "merchant_category": merchant_category,
                "payment_method": payment_method,
                "city": city,
                "device_type": device_type,
                "hour_of_day": hour_of_day,
                "is_international": is_international
            }

            progress_placeholder = st.empty()
            try:
                with progress_placeholder.container():
                    st.info("1️⃣ New transaction entered")
                    time.sleep(0.4)
                    st.info("2️⃣ Sending to API")
                    time.sleep(0.4)
                    st.info("3️⃣ Running fraud and category scoring")
                    time.sleep(0.5)

                result = post_json("/transactions/ingest", payload)
                st.session_state.last_result = result
                progress_placeholder.empty()
                st.success("✅ Live transaction processed")
                st.cache_data.clear()

            except Exception as e:
                progress_placeholder.empty()
                st.error(f"Processing failed: {e}")

    if st.session_state.last_result:
        result = st.session_state.last_result
        fraud_prob = float(result.get("fraud_probability", 0.0))

        st.markdown("### Latest Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Final Prediction", result.get("fraud_label", "N/A"))
        r2.metric("Final Fraud Score", f"{fraud_prob:.2%}")
        r3.metric("Predicted Spending Category", result.get("predicted_category", "Unknown"))

        st.markdown(f"**Risk Level:** {risk_badge(fraud_prob)}")
        st.markdown(f"**Transaction ID:** `{result.get('transaction_id', 'N/A')}`")

        if fraud_prob >= threshold:
            st.error("🚨 This transaction crossed the fraud alert threshold.")
        else:
            st.success("✅ This transaction is below the fraud alert threshold.")

st.divider()

tab1, tab2, tab3 = st.tabs(["Recent Transactions", "Fraud Analytics", "Spending Insights"])

with tab1:
    st.subheader("Recent Transactions")
    try:
        recent = pd.DataFrame(get_json("/transactions/recent"))

        if not recent.empty:
            display_cols = [
                "transaction_id",
                "customer_id",
                "merchant",
                "type",
                "amount",
                "fraud_probability",
                "fraud_label",
                "risk_level",
                "predicted_category",
                "timestamp",
            ]

            display_cols = [c for c in display_cols if c in recent.columns]
            recent_view = recent[display_cols].copy()

            if "amount" in recent_view.columns:
                recent_view["amount"] = pd.to_numeric(recent_view["amount"], errors="coerce")

            if "fraud_probability" in recent_view.columns:
                recent_view["fraud_probability"] = pd.to_numeric(
                    recent_view["fraud_probability"], errors="coerce"
                )

            styled_recent = (
                style_transactions(recent_view)
                .format({
                    "amount": "${:,.2f}",
                    "fraud_probability": "{:.2%}",
                }, na_rep="—")
            )

            st.dataframe(styled_recent, use_container_width=True, height=420)
        else:
            st.info("No transactions available yet.")

    except Exception as e:
        st.warning(f"Could not load recent transactions: {e}")

with tab2:
    st.subheader("Fraud Analytics")
    try:
        hourly_df = pd.DataFrame(get_json("/fraud/hourly-bars"))
        by_type = pd.DataFrame(get_json("/fraud/by-type"))
        top_merchants = pd.DataFrame(get_json("/fraud/top-merchants"))
        recent = pd.DataFrame(get_json("/transactions/recent"))
    except Exception:
        hourly_df = pd.DataFrame()
        by_type = pd.DataFrame()
        top_merchants = pd.DataFrame()
        recent = pd.DataFrame()

    c1, c2 = st.columns(2)

    with c1:
        if not hourly_df.empty and {"hour", "fraud_count"}.issubset(hourly_df.columns):
            hourly_df["hour_label"] = hourly_df["hour"].apply(lambda x: f"{int(x):02d}:00")
            fig = px.bar(
                hourly_df,
                x="hour_label",
                y="fraud_count",
                title="Fraud Count by Hour of Day",
                color="fraud_count",
                color_continuous_scale="Reds"
            )
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Fraud Count",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)

            peak_hour = hourly_df.loc[hourly_df["fraud_count"].idxmax(), "hour_label"]
            peak_count = int(hourly_df["fraud_count"].max())
            st.caption(f"Peak fraud activity appears around **{peak_hour}** with **{peak_count}** flagged transactions.")
        else:
            st.info("No hourly fraud data available.")

    with c2:
        if not recent.empty and {"amount", "fraud_label"}.issubset(recent.columns):
            recent["amount"] = pd.to_numeric(recent["amount"], errors="coerce")
            fig = px.box(
                recent,
                x="fraud_label",
                y="amount",
                color="fraud_label",
                title="Fraud vs Non-Fraud Amount Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No amount distribution data available.")

    c3, c4 = st.columns(2)

    with c3:
        if not by_type.empty:
            fig = px.bar(
                by_type.sort_values("fraud_count", ascending=True),
                x="fraud_count",
                y="type",
                orientation="h",
                title="Fraud Count by Transaction Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No transaction type fraud data available.")

    with c4:
        if not top_merchants.empty:
            fig = px.bar(
                top_merchants.sort_values("flagged_amount", ascending=True),
                x="flagged_amount",
                y="merchant",
                orientation="h",
                title="Top Flagged Merchants"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No flagged merchant data available.")
with tab3:
    st.subheader("Spending Insights")

    try:
        spending = pd.DataFrame(get_json("/spending/summary"))
        treemap_df = pd.DataFrame(get_json("/spending/merchant-treemap"))
    except Exception:
        spending = pd.DataFrame()
        treemap_df = pd.DataFrame()

    if not spending.empty:
        spending.columns = spending.columns.str.strip()
        if "category" not in spending.columns and "predicted_category" in spending.columns:
            spending = spending.rename(columns={"predicted_category": "category"})

    if not treemap_df.empty:
        treemap_df.columns = treemap_df.columns.str.strip()
        if "predicted_category" not in treemap_df.columns and "category" in treemap_df.columns:
            treemap_df = treemap_df.rename(columns={"category": "predicted_category"})

    c1, c2 = st.columns(2)

    with c1:
        if not spending.empty and "category" in spending.columns:
            fig1 = px.bar(
                spending.sort_values("total_amount", ascending=True),
                x="total_amount",
                y="category",
                orientation="h",
                title="Spend by Category"
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("No spending data available.")

    with c2:
        if not spending.empty and "category" in spending.columns:
            fig2 = px.bar(
                spending.sort_values("transaction_count", ascending=False),
                x="category",
                y="transaction_count",
                title="Transactions by Category"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No category count data available.")

    c3, c4 = st.columns(2)

    with c3:
        if not treemap_df.empty and {"predicted_category", "merchant", "total_amount"}.issubset(treemap_df.columns):
            fig3 = px.treemap(
                treemap_df,
                path=["predicted_category", "merchant"],
                values="total_amount",
                title="Merchant Mix Within Spending Categories"
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No merchant-category data available.")

    with c4:
        if not spending.empty and {"category", "fraud_rate"}.issubset(spending.columns):
            fig4 = px.bar(
                spending.sort_values("fraud_rate", ascending=False),
                x="category",
                y="fraud_rate",
                title="Fraud Rate by Spending Category"
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No fraud-by-category data available.")