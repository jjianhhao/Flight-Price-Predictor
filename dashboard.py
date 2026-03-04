import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0f1117; }
    [data-testid="stSidebar"] { background-color: #1a1d27; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #7c9ef8; }
    .metric-label { font-size: 0.85rem; color: #8b92a5; margin-top: 4px; }
    .section-header {
        font-size: 1.4rem; font-weight: 600; color: #e2e8f0;
        border-left: 4px solid #7c9ef8; padding-left: 12px;
        margin: 28px 0 16px 0;
    }
    .insight-box {
        background: #1a1d27;
        border: 1px solid #2e3250;
        border-left: 4px solid #7c9ef8;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #c9d1e0;
        font-size: 0.92rem;
    }
    .buy-badge {
        background: #1a3a2a; color: #4ade80;
        border: 1px solid #4ade80; border-radius: 20px;
        padding: 4px 16px; font-weight: 700; font-size: 1.1rem;
    }
    .wait-badge {
        background: #3a1a1a; color: #f87171;
        border: 1px solid #f87171; border-radius: 20px;
        padding: 4px 16px; font-weight: 700; font-size: 1.1rem;
    }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .stSelectbox label, .stSlider label, .stRadio label { color: #c9d1e0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Data & Model Loading ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Data/Cleaned/cleaned_flight_data.csv", parse_dates=["booking_date", "departure_date"])
    df["route"] = df["origin"] + "-" + df["destination"]
    df["days_to_departure"] = (df["departure_date"] - df["booking_date"]).dt.days
    df["departure_month"] = df["departure_date"].dt.month
    df["departure_dow"] = df["departure_date"].dt.day_name()
    df["season"] = df["departure_month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
    })
    df["month_name"] = df["departure_date"].dt.strftime("%b")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("models/final_random_forest.pkl")
    metadata = joblib.load("models/final_model_metadata.pkl")
    return model, metadata

df = load_data()
model, metadata = load_model()
feature_columns = metadata["features"]

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PALETTE = px.colors.qualitative.Plotly

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ Flight Price Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview", "Price Drivers", "Model Performance", "Buy vs Wait"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"- {len(df):,} flights")
    st.markdown(f"- {df['route'].nunique()} routes")
    st.markdown(f"- {df['airline'].nunique()} airlines")
    st.markdown(f"- 1 year of data")
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown(f"- Random Forest Regressor")
    st.markdown(f"- R² = {metadata['metrics']['R2']:.3f}")
    st.markdown(f"- RMSE = {metadata['metrics']['RMSE']:.3f} (log)")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("# ✈️ Flight Price Predictor — Dashboard")
    st.markdown("An end-to-end ML project predicting flight prices across 4 routes and 5 airlines.")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        ("$362", "Avg Price"),
        ("$300", "Median Price"),
        ("$50 – $3,094", "Price Range"),
        ("79.6%", "Model R²"),
        ("1,809", "Flights Analysed"),
    ]
    for col, (val, label) in zip([col1, col2, col3, col4, col5], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Price distribution + route breakdown
    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown('<div class="section-header">Price Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df, x="price", nbins=60, color_discrete_sequence=["#7c9ef8"],
            labels={"price": "Ticket Price (USD)"},
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(gridcolor="#2e3250"), yaxis=dict(gridcolor="#2e3250"),
        )
        fig.add_vline(x=df["price"].mean(), line_dash="dash", line_color="#f59e0b",
                      annotation_text=f"Mean ${df['price'].mean():.0f}", annotation_font_color="#f59e0b")
        fig.add_vline(x=df["price"].median(), line_dash="dot", line_color="#4ade80",
                      annotation_text=f"Median ${df['price'].median():.0f}", annotation_font_color="#4ade80")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Avg Price by Route</div>', unsafe_allow_html=True)
        route_avg = df.groupby("route")["price"].mean().sort_values(ascending=True).reset_index()
        fig2 = px.bar(
            route_avg, x="price", y="route", orientation="h",
            color="price", color_continuous_scale="Blues",
            labels={"price": "Avg Price (USD)", "route": ""},
            template="plotly_dark",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(gridcolor="#2e3250"), yaxis=dict(gridcolor="#2e3250"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Key findings
    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)
    findings = [
        "Route is the strongest price driver — JFK-LAX averages ~3× more than LHR-CDG.",
        "Booking early (90–180 days out) consistently yields the lowest prices.",
        "Summer has the highest median prices; Winter and Fall are the cheapest seasons.",
        "Holiday departures are statistically more expensive (p = 0.032).",
        "Random Forest outperforms all linear models, achieving R² = 0.796 in log space.",
        "Airline choice significantly affects price: United/Delta charge more than Ryanair/Indigo.",
    ]
    col_f1, col_f2 = st.columns(2)
    for i, f in enumerate(findings):
        col = col_f1 if i % 2 == 0 else col_f2
        with col:
            st.markdown(f'<div class="insight-box">💡 {f}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRICE DRIVERS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Price Drivers":
    st.markdown("# Price Drivers")
    st.markdown("Explore how each factor statistically influences flight prices.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Route", "Airline", "Season / Month", "Day of Week", "Booking Lead Time", "Stops & Holiday"]
    )

    # ── Route ──
    with tab1:
        col_l, col_r = st.columns([1.2, 1])
        with col_l:
            fig = px.box(
                df, x="route", y="price", color="route",
                color_discrete_sequence=PALETTE,
                labels={"price": "Price (USD)", "route": "Route"},
                template="plotly_dark",
                title="Price Distribution by Route",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              showlegend=False, xaxis=dict(gridcolor="#2e3250"), yaxis=dict(gridcolor="#2e3250"))
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            stats = df.groupby("route")["price"].agg(["mean", "median", "std", "count"]).round(2)
            stats.columns = ["Mean", "Median", "Std Dev", "Count"]
            st.dataframe(stats.style.background_gradient(cmap="Blues", subset=["Mean"]), use_container_width=True)
            st.markdown('<div class="insight-box">ANOVA p-value: 3.31×10⁻²⁸⁷ — Route is the most statistically significant price driver in the dataset. JFK-LAX (long-haul, US domestic) commands the highest fares, while LHR-CDG (short-haul, European) is the cheapest.</div>', unsafe_allow_html=True)

    # ── Airline ──
    with tab2:
        col_l, col_r = st.columns([1.2, 1])
        with col_l:
            fig = px.box(
                df, x="airline", y="price", color="airline",
                color_discrete_sequence=PALETTE,
                labels={"price": "Price (USD)", "airline": "Airline"},
                template="plotly_dark",
                title="Price Distribution by Airline",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              showlegend=False, xaxis=dict(gridcolor="#2e3250"), yaxis=dict(gridcolor="#2e3250"))
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            stats = df.groupby("airline")["price"].agg(["mean", "median", "std", "count"]).round(2)
            stats.columns = ["Mean", "Median", "Std Dev", "Count"]
            st.dataframe(stats.sort_values("Mean", ascending=False).style.background_gradient(cmap="Blues", subset=["Mean"]), use_container_width=True)
            st.markdown('<div class="insight-box">ANOVA p-value: 1.40×10⁻²⁸⁷ — Airline is equally significant as route. United and Delta (full-service carriers) charge substantially more than Ryanair and Indigo (low-cost carriers).</div>', unsafe_allow_html=True)

    # ── Season / Month ──
    with tab3:
        col_l, col_r = st.columns(2)
        with col_l:
            season_order = ["Winter", "Spring", "Summer", "Fall"]
            fig = px.box(
                df, x="season", y="price", color="season",
                category_orders={"season": season_order},
                color_discrete_sequence=["#60a5fa", "#34d399", "#f59e0b", "#f87171"],
                labels={"price": "Price (USD)", "season": "Season"},
                template="plotly_dark",
                title="Price by Season",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              showlegend=False, xaxis=dict(gridcolor="#2e3250"), yaxis=dict(gridcolor="#2e3250"))
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            month_avg = df.groupby("month_name")["price"].median().reindex(MONTH_ORDER).reset_index()
            month_avg.columns = ["Month", "Median Price"]
            fig2 = px.line(
                month_avg, x="Month", y="Median Price",
                markers=True, color_discrete_sequence=["#7c9ef8"],
                template="plotly_dark",
                title="Median Price by Departure Month",
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               xaxis=dict(gridcolor="#2e3250"), yaxis=dict(gridcolor="#2e3250"))
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="insight-box">ANOVA p-value: 0.00255 (season), 0.00756 (month) — Summer peaks in price. June–August and October–December show elevated fares. February–April tend to be the cheapest months to fly.</div>', unsafe_allow_html=True)

    # ── Day of Week ──
    with tab4:
        dow_avg = df.groupby("departure_dow")["price"].median().reindex(DOW_ORDER).reset_index()
        dow_avg.columns = ["Day", "Median Price"]
        fig = px.bar(
            dow_avg, x="Day", y="Median Price",
            color="Median Price", color_continuous_scale="Blues",
            template="plotly_dark",
            title="Median Price by Departure Day of Week",
            labels={"Median Price": "Median Price (USD)"},
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          coloraxis_showscale=False, xaxis=dict(gridcolor="#2e3250"), yaxis=dict(gridcolor="#2e3250"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">ANOVA p-value: 2.97×10⁻⁸ — Fridays and weekends carry slightly higher median prices. Midweek departures (Mon–Wed) are generally cheaper. The effect is statistically significant but moderate in magnitude.</div>', unsafe_allow_html=True)

    # ── Booking Lead Time ──
    with tab5:
        df_dbd = df[df["days_to_departure"] >= 0].copy()
        df_dbd["lead_bin"] = pd.cut(
            df_dbd["days_to_departure"],
            bins=[0, 7, 14, 30, 60, 90, 180, 365],
            labels=["0–7d", "8–14d", "15–30d", "31–60d", "61–90d", "91–180d", "181–365d"],
        )
        lead_avg = df_dbd.groupby("lead_bin", observed=True)["price"].median().reset_index()
        lead_avg.columns = ["Booking Window", "Median Price"]
        fig = px.line(
            lead_avg, x="Booking Window", y="Median Price",
            markers=True, color_discrete_sequence=["#f59e0b"],
            template="plotly_dark",
            title="Median Price vs Booking Lead Time",
            labels={"Median Price": "Median Price (USD)"},
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(gridcolor="#2e3250"), yaxis=dict(gridcolor="#2e3250"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">ANOVA p-value: 1.13×10⁻²⁵ — The earlier you book, the cheaper the fare. Flights booked 0–7 days out have the highest median prices. The sweet spot is 91–180 days before departure.</div>', unsafe_allow_html=True)

    # ── Stops & Holiday ──
    with tab6:
        col_l, col_r = st.columns(2)
        with col_l:
            fig = px.box(
                df, x=df["stops"].map({0: "Non-stop", 1: "1 Stop"}), y="price",
                color=df["stops"].map({0: "Non-stop", 1: "1 Stop"}),
                color_discrete_sequence=["#7c9ef8", "#f59e0b"],
                labels={"price": "Price (USD)", "x": ""},
                template="plotly_dark",
                title="Price by Number of Stops",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              showlegend=False, xaxis=dict(gridcolor="#2e3250"), yaxis=dict(gridcolor="#2e3250"))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<div class="insight-box">Welch t-test p-value: 0.536 — No statistically significant price difference between non-stop and 1-stop flights in this dataset.</div>', unsafe_allow_html=True)
        with col_r:
            fig2 = px.box(
                df[df["departure_date"].dt.month.isin(range(1, 13))],
                x=df["stops"].map({0: "Non-Holiday", 1: "Holiday"}).reindex(df.index),
                y="price",
                color=df["stops"].map({0: "Non-Holiday", 1: "Holiday"}).reindex(df.index),
                color_discrete_sequence=["#34d399", "#f87171"],
                labels={"price": "Price (USD)", "x": ""},
                template="plotly_dark",
                title="Price: Holiday vs Non-Holiday",
            )
            # Use actual holiday flag from features file
            feat = pd.read_csv("Data/Cleaned/flight_features.csv")
            holiday_map = feat["is_holiday"].values
            df_h = df.copy()
            df_h = df_h.iloc[:len(holiday_map)].copy()
            df_h["is_holiday"] = holiday_map
            fig2 = px.box(
                df_h, x=df_h["is_holiday"].map({0: "Non-Holiday", 1: "Holiday"}), y="price",
                color=df_h["is_holiday"].map({0: "Non-Holiday", 1: "Holiday"}),
                color_discrete_sequence=["#34d399", "#f87171"],
                labels={"price": "Price (USD)", "x": ""},
                template="plotly_dark",
                title="Price: Holiday vs Non-Holiday",
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               showlegend=False, xaxis=dict(gridcolor="#2e3250"), yaxis=dict(gridcolor="#2e3250"))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('<div class="insight-box">Welch t-test p-value: 0.032 — Holiday departures are statistically more expensive with greater price variability, reflecting demand spikes during peak travel periods.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown("# Model Performance")
    st.markdown("Comparison of all regression models trained in log space.")

    # Model comparison table (from notebook outputs)
    model_results = pd.DataFrame([
        {"Model": "Linear Regression", "MAE": 0.2810, "RMSE": 0.3601, "R²": 0.620},
        {"Model": "Ridge Regression",  "MAE": 0.2810, "RMSE": 0.3601, "R²": 0.620},
        {"Model": "Lasso Regression",  "MAE": 0.2813, "RMSE": 0.3604, "R²": 0.620},
        {"Model": "Random Forest",     "MAE": 0.1921, "RMSE": 0.2706, "R²": 0.789},
        {"Model": "Random Forest (Tuned)", "MAE": metadata["metrics"]["MAE"], "RMSE": float(metadata["metrics"]["RMSE"]), "R²": metadata["metrics"]["R2"]},
    ])

    col1, col2, col3 = st.columns(3)
    best = model_results.loc[model_results["R²"].idxmax()]
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{best['R²']:.3f}</div>
            <div class="metric-label">Best R² (Random Forest Tuned)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{best['RMSE']:.3f}</div>
            <div class="metric-label">Best RMSE (log space)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{best['MAE']:.3f}</div>
            <div class="metric-label">Best MAE (log space)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    col_l, col_r = st.columns([1.3, 1])

    with col_l:
        st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
        fig = go.Figure()
        colors = {"MAE": "#7c9ef8", "RMSE": "#f59e0b", "R²": "#4ade80"}
        metrics_to_plot = ["MAE", "RMSE"]
        for metric in metrics_to_plot:
            fig.add_trace(go.Bar(
                name=metric,
                x=model_results["Model"],
                y=model_results[metric],
                marker_color=colors[metric],
            ))
        fig.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#2e3250", tickangle=-20),
            yaxis=dict(gridcolor="#2e3250", title="Score (log space)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">R² Comparison</div>', unsafe_allow_html=True)
        fig2 = px.bar(
            model_results, x="R²", y="Model", orientation="h",
            color="R²", color_continuous_scale="Blues",
            range_x=[0, 1],
            template="plotly_dark",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#2e3250"),
            yaxis=dict(gridcolor="#2e3250"),
            margin=dict(t=20, b=10),
        )
        fig2.add_vline(x=0.8, line_dash="dash", line_color="#4ade80",
                       annotation_text="0.8 threshold", annotation_font_color="#4ade80")
        st.plotly_chart(fig2, use_container_width=True)

    # Full table
    st.markdown('<div class="section-header">Full Results Table</div>', unsafe_allow_html=True)
    styled = model_results.style \
        .background_gradient(cmap="Blues", subset=["R²"]) \
        .background_gradient(cmap="Reds_r", subset=["RMSE", "MAE"]) \
        .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"})
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Feature importance
    st.markdown('<div class="section-header">Feature Importance (Random Forest Tuned)</div>', unsafe_allow_html=True)
    importances = model_results  # placeholder — compute from model
    feat_imp = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig3 = px.bar(
        feat_imp, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
        template="plotly_dark",
        labels={"Importance": "Feature Importance", "Feature": ""},
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="#2e3250"),
        yaxis=dict(gridcolor="#2e3250"),
        height=420,
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig3, use_container_width=True)

    col_ins1, col_ins2 = st.columns(2)
    with col_ins1:
        st.markdown('<div class="insight-box">Tree-based models capture non-linear relationships and feature interactions that linear models miss — explaining the ~17 percentage point R² improvement.</div>', unsafe_allow_html=True)
    with col_ins2:
        st.markdown('<div class="insight-box">All metrics are computed in log space (log1p transformed target). Predictions are inverse-transformed (expm1) for business interpretation.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — BUY VS WAIT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Buy vs Wait":
    st.markdown("# Buy vs Wait Strategy")
    st.markdown("Use the trained model to decide whether to book now or wait for a better price.")

    st.markdown('<div class="section-header">Configure Your Flight</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        route = st.selectbox("Route", ["JFK-LAX", "LHR-CDG", "SYD-MEL", "BOM-DEL"])
        airline = st.selectbox("Airline", ["United", "Ryanair", "Qantas", "Indigo", "Delta"])
    with col2:
        month = st.slider("Departure Month", 1, 12, 7, format="%d")
        stops = st.radio("Stops", [0, 1], format_func=lambda x: "Non-stop" if x == 0 else "1 Stop")
    with col3:
        is_weekend = st.radio("Departure Day", ["Weekday", "Weekend"]) == "Weekend"
        is_holiday = st.radio("Holiday Departure?", ["No", "Yes"]) == "Yes"

    # Map to feature columns
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Fall", 10: "Fall", 11: "Fall"}
    season = season_map[month]

    days_range = np.arange(1, 91)
    rows = []
    for d in days_range:
        row = {
            "stops": stops,
            "days_to_departure_bin": d // 7,
            "departure_month": month,
            "is_weekend": int(is_weekend),
            "season_Spring": int(season == "Spring"),
            "season_Summer": int(season == "Summer"),
            "season_Winter": int(season == "Winter"),
            "is_holiday": int(is_holiday),
            "route_JFK-LAX": int(route == "JFK-LAX"),
            "route_LHR-CDG": int(route == "LHR-CDG"),
            "route_SYD-MEL": int(route == "SYD-MEL"),
            "airline_INDIGO": int(airline.upper() == "INDIGO"),
            "airline_QANTAS": int(airline.upper() == "QANTAS"),
            "airline_RYANAIR": int(airline.upper() == "RYANAIR"),
            "airline_UNITED": int(airline.upper() == "UNITED"),
        }
        rows.append(row)

    future_df = pd.DataFrame(rows)[feature_columns]
    future_df["predicted_log_price"] = model.predict(future_df)
    future_df["predicted_price"] = np.expm1(future_df["predicted_log_price"])
    future_df["days_to_departure"] = days_range

    threshold_price = np.percentile(future_df["predicted_price"], 20)
    future_df["decision"] = np.where(future_df["predicted_price"] <= threshold_price, "Buy", "Wait")

    # Current day recommendation
    st.markdown('<div class="section-header">Recommendation for Today</div>', unsafe_allow_html=True)
    today_days = st.slider("Days until departure (today)", 1, 90, 30)
    today_row = future_df[future_df["days_to_departure"] == today_days].iloc[0]
    decision = today_row["decision"]
    price_now = today_row["predicted_price"]
    min_price = future_df["predicted_price"].min()

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        badge = f'<span class="buy-badge">BUY NOW</span>' if decision == "Buy" else f'<span class="wait-badge">WAIT</span>'
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:0.85rem;color:#8b92a5;margin-bottom:8px">Recommendation</div>
            {badge}
        </div>""", unsafe_allow_html=True)
    with col_d2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">${price_now:.0f}</div>
            <div class="metric-label">Predicted Price Now</div>
        </div>""", unsafe_allow_html=True)
    with col_d3:
        saving = price_now - min_price
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">${saving:.0f}</div>
            <div class="metric-label">Potential Saving (vs best window)</div>
        </div>""", unsafe_allow_html=True)

    # Price trend chart
    st.markdown('<div class="section-header">Predicted Price Trend (Next 90 Days)</div>', unsafe_allow_html=True)

    fig = go.Figure()
    buy_mask = future_df["decision"] == "Buy"
    wait_mask = future_df["decision"] == "Wait"

    fig.add_trace(go.Scatter(
        x=future_df[wait_mask]["days_to_departure"],
        y=future_df[wait_mask]["predicted_price"],
        mode="lines+markers", name="Wait",
        line=dict(color="#f87171", width=2),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=future_df[buy_mask]["days_to_departure"],
        y=future_df[buy_mask]["predicted_price"],
        mode="markers", name="Buy Window",
        marker=dict(color="#4ade80", size=10, symbol="star"),
    ))
    fig.add_vline(
        x=today_days, line_dash="dash", line_color="#f59e0b",
        annotation_text=f"Today ({today_days}d)", annotation_font_color="#f59e0b",
    )
    fig.add_hline(
        y=threshold_price, line_dash="dot", line_color="#4ade80",
        annotation_text=f"Buy threshold ${threshold_price:.0f}", annotation_font_color="#4ade80",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Days to Departure", gridcolor="#2e3250"),
        yaxis=dict(title="Predicted Price (USD)", gridcolor="#2e3250"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="insight-box">
    <b>Decision Rule:</b> "Buy" is recommended when the predicted price falls within the bottom 20% of all predicted prices across the 90-day window.
    The stepwise pattern reflects the model's use of binned booking windows — a realistic representation of how airlines update fares at discrete intervals.
    </div>""", unsafe_allow_html=True)
