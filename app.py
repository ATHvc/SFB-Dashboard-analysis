import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="SFB Financial Dashboard", layout="wide")
st.title("SFB Financial Intelligence Dashboard")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #020617 100%);
    color: #f8fafc;
}

h1 {
    font-size: 42px !important;
    font-weight: 800 !important;
    color: #f8fafc !important;
}

h2, h3 {
    color: #e5e7eb !important;
    font-weight: 700 !important;
}

[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.85);
    padding: 18px;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.25);
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-size: 30px !important;
    font-weight: 800 !important;
}

.stTabs [data-baseweb="tab"] {
    background-color: #1e293b;
    border-radius: 12px 12px 0 0;
    padding: 12px 18px;
    color: #cbd5e1;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #2563eb !important;
    color: white !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.25);
}

.stAlert {
    border-radius: 14px;
}

.stButton button, .stDownloadButton button {
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    color: white;
    border-radius: 12px;
    border: none;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

df = pd.read_csv("cleaned_sfb_data.csv")
yearly_df = pd.read_csv("sfb_yearly_timeseries.csv")

df["Bank"] = df["Bank"].astype(str).str.strip()
df["Metric"] = df["Metric"].astype(str).str.strip()
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df.dropna(subset=["Value"])

yearly_df["Bank"] = yearly_df["Bank"].astype(str).str.strip()
yearly_df["Metric"] = yearly_df["Metric"].astype(str).str.strip()
yearly_df["Value"] = pd.to_numeric(yearly_df["Value"], errors="coerce")
yearly_df = yearly_df.dropna(subset=["Value"])

year_map = {"FY24": 2024, "FY25": 2025, "FY26": 2026}
yearly_df["Year_Num"] = yearly_df["Year"].map(year_map)

# ML model preparation
model_df = df.pivot_table(
    index="Bank",
    columns="Metric",
    values="Value",
    aggfunc="mean"
).reset_index()

model_df = model_df.fillna(0)

features = model_df.drop(columns=["Bank"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
model_df["Cluster"] = kmeans.fit_predict(X_scaled)

cluster_label_map = {
    0: "Weak / Risky",
    1: "Stable / High Performing",
    2: "Outlier / Needs Review"
}
model_df["Cluster_Label"] = model_df["Cluster"].map(cluster_label_map)

iso = IsolationForest(contamination=0.15, random_state=42)
model_df["Anomaly"] = iso.fit_predict(X_scaled)
model_df["Anomaly_Label"] = model_df["Anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})

def minmax(series):
    if series.max() == series.min():
        return series * 0
    return (series - series.min()) / (series.max() - series.min())

score_df = model_df.copy()

score_df["Growth_Score"] = 0
score_df["Profitability_Score"] = 0
score_df["Risk_Score"] = 70

if "Revenue growth (YoY)" in score_df.columns:
    score_df["Growth_Score"] = minmax(score_df["Revenue growth (YoY)"]) * 100

profit_cols = [
    c for c in ["Gross margin", "EBITDA margin", "PBT margin", "ROA", "ROE"]
    if c in score_df.columns
]

if profit_cols:
    score_df["Profitability_Score"] = (
        score_df[profit_cols].apply(minmax).mean(axis=1) * 100
    )

if "Debt equity ratio" in score_df.columns:
    score_df["Risk_Score"] = (
        1 - minmax(score_df["Debt equity ratio"])
    ) * 100

score_df["Final_Bank_Score"] = (
    0.35 * score_df["Growth_Score"] +
    0.45 * score_df["Profitability_Score"] +
    0.20 * score_df["Risk_Score"]
)

score_df = score_df.sort_values("Final_Bank_Score", ascending=False)

selected_banks = st.multiselect(
    "Select Banks",
    options=sorted(df["Bank"].unique()),
    default=sorted(df["Bank"].unique())
)

selected_metric = st.selectbox(
    "Select Metric",
    options=sorted(df["Metric"].unique())
)

filtered = df[
    (df["Bank"].isin(selected_banks)) &
    (df["Metric"] == selected_metric)
].copy()

filtered = filtered.sort_values("Value", ascending=False)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Overview",
        "Bank Comparison",
        "Rankings",
        "Raw Data",
        "ML / AI Insights",
        "Forecasting"
    ]
)

with tab1:
    st.subheader("Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Selected Banks", len(selected_banks))
    col2.metric("Average Value", round(filtered["Value"].mean(), 4))

    if not filtered.empty:
        col3.metric("Top Bank", filtered.iloc[0]["Bank"])
        col4.metric("Bottom Bank", filtered.iloc[-1]["Bank"])

    fig = px.bar(
        filtered,
        x="Bank",
        y="Value",
        color="Bank",
        title=f"{selected_metric} Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Bank Comparison")

    heatmap_df = df[df["Bank"].isin(selected_banks)].pivot_table(
        index="Bank",
        columns="Metric",
        values="Value",
        aggfunc="mean"
    )

    fig_heatmap = px.imshow(
        heatmap_df,
        text_auto=True,
        aspect="auto",
        title="Bank vs Metric Heatmap"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab3:
    st.subheader("Rankings")

    st.dataframe(filtered, use_container_width=True)

    fig_rank = px.bar(
        filtered,
        x="Value",
        y="Bank",
        orientation="h",
        color="Bank",
        title=f"Ranking by {selected_metric}"
    )
    st.plotly_chart(fig_rank, use_container_width=True)

with tab4:
    st.subheader("Raw Data")

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        csv,
        "sfb_financial_dashboard_data.csv",
        "text/csv"
    )

with tab5:
    st.subheader("ML / AI-Assisted Bank Analysis")

    investment_amount = st.number_input(
        "Enter Investment Amount in Crores",
        min_value=1,
        value=100
    )

    best_bank = score_df.iloc[0]["Bank"]
    best_score = score_df.iloc[0]["Final_Bank_Score"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Best Bank", best_bank)
    col2.metric("Best Score", round(best_score, 2))
    col3.metric("Investment Amount", f"₹{investment_amount} Cr")

    st.success(
        f"Based on KMeans clustering, Isolation Forest anomaly detection, "
        f"and financial scoring, **{best_bank}** appears to be the strongest bank "
        f"with a score of **{best_score:.2f}/100**."
    )

    st.subheader("ML Model Output")

    st.dataframe(
        score_df[
            [
                "Bank",
                "Growth_Score",
                "Profitability_Score",
                "Risk_Score",
                "Final_Bank_Score",
                "Cluster",
                "Cluster_Label",
                "Anomaly_Label"
            ]
        ],
        use_container_width=True
    )

    st.subheader("Suggested Allocation")

    top_3 = score_df.head(3).copy()
    total_score = top_3["Final_Bank_Score"].sum()

    top_3["Suggested_Allocation_Cr"] = (
        top_3["Final_Bank_Score"] / total_score * investment_amount
    )

    st.dataframe(
        top_3[["Bank", "Final_Bank_Score", "Suggested_Allocation_Cr"]],
        use_container_width=True
    )

    fig_score = px.bar(
        score_df,
        x="Bank",
        y="Final_Bank_Score",
        color="Anomaly_Label",
        title="Final Bank Score with Anomaly Detection"
    )
    st.plotly_chart(fig_score, use_container_width=True)

    fig_cluster = px.scatter(
        score_df,
        x="Growth_Score",
        y="Profitability_Score",
        color="Cluster_Label",
        size="Final_Bank_Score",
        hover_name="Bank",
        title="KMeans Clustering: Growth vs Profitability"
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.warning(
        "This is an ML-assisted analytical overview based on available sample data. "
        "It is not financial advice."
    )

with tab6:
    st.subheader("Time Series Forecasting")

    forecast_bank = st.selectbox(
        "Select Bank for Forecasting",
        sorted(yearly_df["Bank"].unique())
    )

    forecast_metric = st.selectbox(
        "Select Metric for Forecasting",
        sorted(yearly_df["Metric"].unique())
    )

    forecast_data = yearly_df[
        (yearly_df["Bank"] == forecast_bank) &
        (yearly_df["Metric"] == forecast_metric)
    ].sort_values("Year_Num")

    st.dataframe(
        forecast_data[["Bank", "Year", "Metric", "Value"]],
        use_container_width=True
    )

    if len(forecast_data) >= 3:
        X = forecast_data[["Year_Num"]]
        y = forecast_data["Value"]

        model = LinearRegression()
        model.fit(X, y)

        future_year = 2027
        predicted_value = model.predict([[future_year]])[0]

        col1, col2, col3 = st.columns(3)

        col1.metric("Selected Bank", forecast_bank)
        col2.metric("Selected Metric", forecast_metric)
        col3.metric("Predicted FY27", round(predicted_value, 2))

        forecast_plot = forecast_data.copy()

        forecast_plot = pd.concat([
            forecast_plot,
            pd.DataFrame({
                "Bank": [forecast_bank],
                "Year": ["FY27 Forecast"],
                "Metric": [forecast_metric],
                "Value": [predicted_value],
                "Year_Num": [future_year]
            })
        ])

        fig_forecast = px.line(
            forecast_plot,
            x="Year",
            y="Value",
            markers=True,
            title=f"{forecast_metric} Forecast for {forecast_bank}"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        last_value = forecast_data["Value"].iloc[-1]
        change = predicted_value - last_value
        change_pct = (change / last_value * 100) if last_value != 0 else 0

        if change > 0:
            explanation = (
                f"{forecast_bank} shows an improving trend in {forecast_metric}. "
                f"The Linear Regression model predicts FY27 value to increase by "
                f"approximately {change_pct:.2f}% compared to FY26."
            )
        else:
            explanation = (
                f"{forecast_bank} shows a declining trend in {forecast_metric}. "
                f"The Linear Regression model predicts FY27 value to decrease by "
                f"approximately {abs(change_pct):.2f}% compared to FY26."
            )

        st.success(explanation)

    else:
        st.warning("Not enough yearly data available.")