
import streamlit as st
import streamlit_shadcn_ui as ui
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from agent import run_investigation
from prompt_interpreter import interpret_prompt


# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="AnomalyX | Intelligence Console",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================================
# CUSTOM DESIGN
# =====================================================
st.markdown("""
<style>
.block-container {
    max-width: 1450px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}
.ax-header {
    margin-bottom: 1.5rem;
}
.ax-brand {
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #F8FAFC;
}
.ax-subtitle {
    color: #94A3B8;
    font-size: 0.9rem;
    margin-top: 0.15rem;
}
.ax-section {
    color: #F1F5F9;
    font-size: 1.2rem;
    font-weight: 650;
    margin: 1.6rem 0 0.9rem;
}
.ax-kpi {
    border-radius: 16px;
    padding: 22px 24px;
    min-height: 145px;
    border: 1px solid;
}
.ax-kpi-label {
    font-size: 0.84rem;
    margin-bottom: 14px;
}
.ax-kpi-value {
    font-size: 2.25rem;
    font-weight: 750;
    color: #FFFFFF;
    line-height: 1.1;
}
.ax-kpi-accent {
    width: 54px;
    height: 4px;
    border-radius: 8px;
    margin-top: 18px;
}
[data-testid="stMetricValue"] {
    color: var(--metric-color, #F1F5F9);
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    features = pd.read_csv("ecommerce_features.csv")
    anomalies = pd.read_csv("lof_detected_anomalies.csv")

    features["date"] = pd.to_datetime(features["date"])
    anomalies["date"] = pd.to_datetime(anomalies["date"])

    anomalies = anomalies.sort_values(
        "lof_score",
        ascending=True
    ).reset_index(drop=True)

    anomalies["Anomaly Rank"] = anomalies.index + 1

    return features, anomalies


features, anomalies = load_data()


# =====================================================
# REUSABLE COMPONENTS
# =====================================================
def kpi_card(label, value, background, border, accent):
    st.markdown(
        f"""
        <div class="ax-kpi"
             style="background:{background};border-color:{border}">
            <div class="ax-kpi-label" style="color:#CBD5E1">
                {label}
            </div>
            <div class="ax-kpi-value">{value}</div>
            <div class="ax-kpi-accent"
                 style="background:{accent}"></div>
        </div>
        """,
        unsafe_allow_html=True
    )


def style_chart(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="#172338",
        plot_bgcolor="#172338",
        font=dict(color="#CBD5E1"),
        margin=dict(l=10, r=15, t=15, b=15),
        xaxis=dict(
            gridcolor="#334155",
            zeroline=False
        ),
        yaxis=dict(
            gridcolor="#334155",
            zeroline=False
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            y=1.12
        )
    )
    return fig



# =====================================================
# AI INVESTIGATION REPORT
# =====================================================
def extract_final_report(result):
    """Extract the final investigation report from LangGraph state."""

    for message in reversed(result.get("messages", [])):

        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content", "")

        elif isinstance(message, str):
            continue

        else:
            role = getattr(message, "role", None)
            role = role or getattr(message, "type", None)
            content = getattr(message, "content", "")

        if role in ("assistant", "ai") and content:
            return str(content)

    return None


def display_investigation_report(result, product_id, date):

    status = result.get("investigation_status", "unknown")
    tools = result.get("executed_tools", [])

    st.markdown("### AI Investigation Report")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Product", product_id)

    with c2:
        st.metric("Investigation Status", status.title())

    with c3:
        st.metric("Tools Executed", len(tools))

    st.caption(f"Investigation date: {date}")

    # ---------------------------------------------
    # VERIFIED NUMERICAL EVIDENCE
    # ---------------------------------------------
    tool_results = result.get("tool_results", {})

    product_context = next(
        (
            value for key, value in tool_results.items()
            if key.startswith("get_product_context:")
        ),
        None
    )

    historical_context = next(
        (
            value for key, value in tool_results.items()
            if key.startswith("get_historical_context:")
        ),
        None
    )

    if product_context and historical_context:

        st.markdown("#### Verified Evidence")
        st.caption("Percentage changes versus the previous 30-day historical average.")
        metrics = [
            ("Units Sold", "units_sold", "avg_units_sold"),
            ("Price", "price", "avg_price"),
            ("Inventory", "inventory", "avg_inventory"),
            ("Revenue", "revenue", "avg_revenue")
        ]

        columns = st.columns(4)

        for column, (label, current_key, avg_key) in zip(
            columns, metrics
        ):
            current = product_context.get(current_key)
            historical = historical_context.get(avg_key)

            with column:
                if (
                    current is not None
                    and historical is not None
                    and historical != 0
                ):
                    change = (
                        (current - historical) / historical
                    ) * 100

                    if label in ("Units Sold", "Revenue"):
                        value_color = "#22C55E" if change > 0 else "#EF4444" if change < 0 else "#F1F5F9"
                    else:
                        value_color = "#F1F5F9"

                    st.markdown(
                        f'<div style="font-size:0.875rem;color:#CBD5E1;">{label}</div>'
                        f'<div style="font-size:2.25rem;font-weight:500;'
                        f'color:{value_color};">{change:+.1f}%</div>',
                        unsafe_allow_html=True
)
    


    # ---------------------------------------------
    # PEER AND PROMOTION COMPARISONS
    # ---------------------------------------------
    peer_context = tool_results.get(
        f"compare_with_peers:{product_id}:{date}"
    )
    promo_context = tool_results.get(
        f"get_promotion_context:{product_id}:{date}"
    )

    if isinstance(peer_context, dict) or isinstance(promo_context, dict):
        st.markdown("#### Peer & Promotion Comparison")

        col1, col2 = st.columns(2)

        with col1:
            if isinstance(peer_context, dict) and "error" not in peer_context:
                product_ratio = peer_context.get("product_demand_ratio")
                peer_ratio = peer_context.get("peer_avg_demand_ratio")

                if product_ratio is not None and peer_ratio is not None:
                    st.metric(
                        "Demand Ratio vs. Peers",
                        f"{product_ratio:.3f}",
                        delta=f"{product_ratio - peer_ratio:+.3f}",
                        help=f"Peer average: {peer_ratio:.3f}"
                    )
                    st.caption(f"Peer average: {peer_ratio:.3f}")

        with col2:
            if isinstance(promo_context, dict) and "error" not in promo_context:
                current_units = promo_context.get("current_units_sold")
                previous_units = promo_context.get("previous_promo_avg_units_sold")
                previous_days = promo_context.get("previous_promo_days")

                if current_units is not None and previous_units is not None:
                    st.metric(
                        "Promotion Units vs. Previous",
                        f"{current_units:,.0f}",
                        delta=f"{current_units - previous_units:+.1f}",
                        help="Difference versus previous promotion-day average"
                    )
                    st.caption(
                        f"Previous promotion average: {previous_units:.1f} units "
                        f"({previous_days} days)"
                    )



    # ---------------------------------------------
    # FINAL AGENT REPORT
    # ---------------------------------------------
    report = extract_final_report(result)

    st.markdown("#### Investigation Findings")

    if report:
        st.markdown(report)
    else:
        st.warning("No final investigation report was returned.")

    # ---------------------------------------------
    # TECHNICAL EXECUTION DETAILS
    # ---------------------------------------------
    with st.expander("Technical Execution Details"):

        st.write("Executed tools:")
        st.write(tools)

        st.write("Tool call count:")
        st.write(result.get("tool_call_count", 0))

        st.write("Missing-information retries:")
        st.write(result.get("missing_retries", 0))

        st.write("Full LangGraph state:")
        st.write(result)



# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="ax-header">
    <div class="ax-brand">AnomalyX</div>
    <div class="ax-subtitle">
        Intelligent Anomaly Detection & AI-Powered Investigation
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================
# MAIN NAVIGATION
# =====================================================
page = st.segmented_control(
    "Navigation",
    options=[
        "Detection Overview",
        "AI Investigation",
        "Technical Details"
    ],
    default="Detection Overview",
    label_visibility="collapsed",
    selection_mode="single",
    width="stretch"
)

st.write("")


# =====================================================
# PAGE 1 — DETECTION OVERVIEW
# =====================================================
if page == "Detection Overview":

    st.markdown(
        '<div class="ax-section">Detection Overview</div>',
        unsafe_allow_html=True
    )

    # -------------------------------------------------
    # GLOBAL PRODUCT FILTER
    # -------------------------------------------------
    selected_product = st.selectbox(
        "Product",
        ["All Products"] + sorted(
            features["product_id"].unique().tolist()
        ),
        index=0
    )

    if selected_product == "All Products":
        filtered_features = features
        filtered_anomalies = anomalies
    else:
        filtered_features = features[
            features["product_id"] == selected_product
        ]
        filtered_anomalies = anomalies[
            anomalies["product_id"] == selected_product
        ]

    # -------------------------------------------------
    # FILTERED KPI CARDS
    # -------------------------------------------------
    k1, k2, k3 = st.columns(3, gap="medium")

    with k1:
        kpi_card(
            "Records Analyzed",
            f"{len(filtered_features):,}",
            "#1E3048", "#365574", "#38BDF8"
        )

    with k2:
        kpi_card(
            "Detected Anomalies",
            f"{len(filtered_anomalies):,}",
            "#30253F", "#55406C", "#A78BFA"
        )

    with k3:
        kpi_card(
            "Products Monitored",
            str(filtered_features["product_id"].nunique()),
            "#183C3D", "#326765", "#2DD4BF"
        )

    # -------------------------------------------------
    # ANOMALY INTELLIGENCE
    # -------------------------------------------------
    st.markdown(
        '<div class="ax-section">Anomaly Intelligence</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.05, 1], gap="large")

    # -------------------------------------------------
    # RANKED ANOMALY TABLE
    # -------------------------------------------------
    with left:
        st.markdown("#### Detected Anomalies")

        display_df = filtered_anomalies[
            ["Anomaly Rank", "product_id", "date"]
        ].copy()

        display_df["date"] = (
            display_df["date"].dt.strftime("%Y-%m-%d")
        )

        display_df = display_df.rename(columns={
            "product_id": "Product",
            "date": "Date"
        })

        st.dataframe(
            display_df,
            hide_index=True,
            width="stretch",
            height=385
        )

    # -------------------------------------------------
    # RIGHT-HAND CHART
    # -------------------------------------------------
    with right:

        if selected_product == "All Products":

            # -----------------------------------------
            # MOST AFFECTED PRODUCTS
            # -----------------------------------------
            st.markdown("#### Most Affected Products")

            counts = (
                filtered_anomalies.groupby("product_id")
                .size()
                .reset_index(name="Anomalies")
                .sort_values(
                    "Anomalies",
                    ascending=False
                )
                .head(10)
            )

            fig = px.bar(
                counts.sort_values("Anomalies"),
                x="Anomalies",
                y="product_id",
                orientation="h",
                labels={"product_id": "Product"}
            )

            fig.update_traces(
                marker_color="#38BDF8",
                marker_line_width=0
            )

            fig = style_chart(fig, height=410)

            st.plotly_chart(
                fig,
                width="stretch",
                config={"displayModeBar": False}
            )

        else:

            # -----------------------------------------
            # PRODUCT DAILY SALES TREND
            # -----------------------------------------
            st.markdown(
                f"#### {selected_product} — Daily Sales Trend"
            )

            sales_candidates = [
                "sales",
                "daily_sales",
                "revenue",
                "units_sold",
                "quantity_sold"
            ]

            sales_col = next(
                (
                    col for col in sales_candidates
                    if col in filtered_features.columns
                ),
                None
            )

            if sales_col is None:
                st.warning(
                    "No daily sales column was found. "
                    "Please check ecommerce_features.csv."
                )

            else:
                product_data = filtered_features.sort_values(
                    "date"
                )

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=product_data["date"],
                        y=product_data[sales_col],
                        mode="lines",
                        name="Daily Sales",
                        line=dict(
                            color="#38BDF8",
                            width=2
                        )
                    )
                )

                # Highlight actual detected anomaly dates
                anomaly_points = product_data.merge(
                    filtered_anomalies[
                        ["product_id", "date"]
                    ].drop_duplicates(),
                    on=["product_id", "date"],
                    how="inner"
                )

                if not anomaly_points.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_points["date"],
                            y=anomaly_points[sales_col],
                            mode="markers",
                            name="Detected Anomaly",
                            marker=dict(
                                color="#F472B6",
                                size=10,
                                symbol="circle",
                                line=dict(
                                    color="#FFFFFF",
                                    width=1
                                )
                            )
                        )
                    )

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title=sales_col.replace(
                        "_", " "
                    ).title()
                )

                fig = style_chart(fig, height=410)

                st.plotly_chart(
                    fig,
                    width="stretch",
                    config={"displayModeBar": False}
                )

    # -------------------------------------------------
    # DETECTION TIMELINE — FIXED
    # -------------------------------------------------
    st.markdown(
        '<div class="ax-section">Detection Timeline</div>',
        unsafe_allow_html=True
    )

    # Define the complete monthly range from the dataset.
    all_months = pd.date_range(
        start=features["date"].min()
        .to_period("M").to_timestamp(),
        end=features["date"].max()
        .to_period("M").to_timestamp(),
        freq="MS"
    )

    # Count anomalies per month, including missing months.
    monthly_counts = (
        filtered_anomalies
        .set_index("date")
        .resample("MS")
        .size()
        if not filtered_anomalies.empty
        else pd.Series(dtype="int64")
    )

    timeline = (
        monthly_counts
        .reindex(all_months, fill_value=0)
        .rename_axis("Month")
        .reset_index(name="Anomalies")
    )

    timeline["Month"] = (
        timeline["Month"].dt.strftime("%Y-%m")
    )

    # Monthly bar chart avoids misleading interpolation.
    fig_timeline = px.bar(
        timeline,
        x="Month",
        y="Anomalies",
        labels={
            "Month": "Month",
            "Anomalies": "Detected Anomalies"
        }
    )

    fig_timeline.update_traces(
        marker_color="#A78BFA",
        marker_line_width=0
    )

    fig_timeline = style_chart(
        fig_timeline,
        height=300
    )

    fig_timeline.update_layout(
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=timeline["Month"].tolist(),
            showgrid=False
        ),
        yaxis=dict(
            gridcolor="#334155",
            rangemode="tozero",
            dtick=1
        ),
        bargap=0.3
    )

    st.plotly_chart(
        fig_timeline,
        width="stretch",
        config={"displayModeBar": False}
    )


# =====================================================
# PAGE 2 — AI INVESTIGATION
# =====================================================
elif page == "AI Investigation":

    st.markdown(
        '<div class="ax-section">AI Investigation Workspace</div>',
        unsafe_allow_html=True
    )

    tab_auto, tab_prompt = st.tabs([
        "Automated Investigation",
        "Ask AnomalyX"
    ])

    # -------------------------------------------------
    # AUTOMATED INVESTIGATION
    # -------------------------------------------------
    with tab_auto:

        top_anomaly = anomalies.iloc[0]

        top_product = str(top_anomaly["product_id"])
        top_date = top_anomaly["date"].strftime("%Y-%m-%d")

        ui.card(
            title="Automated AI Investigation",
            content=(
                "AnomalyX automatically selects the highest-ranked "
                "detected anomaly for AI investigation. "
                "No manual product or date selection is required."
            ),
            key="auto_card"
        )

        st.caption(
            f"Selected anomaly: Rank #1 | "
            f"Product {top_product} | {top_date}"
        )

        
        if ui.button(
            "Run Automated Investigation",
            key="run_auto"
        ):

            with st.spinner(
                f"Investigating {top_product} on {top_date}..."
            ):
                try:
                    result = run_investigation(
                        top_product,
                        top_date
                    )

                    st.session_state["auto_result"] = result
                    st.session_state["auto_product"] = top_product
                    st.session_state["auto_date"] = top_date

                except Exception as e:
                    st.error(f"Investigation failed: {e}")

        # Display the result outside the button condition
        if "auto_result" in st.session_state:

            display_investigation_report(
                result=st.session_state["auto_result"],
                product_id=st.session_state["auto_product"],
                date=st.session_state["auto_date"]
            )



    # -------------------------------------------------
    # ASK ANOMALYX
    # -------------------------------------------------

    with tab_prompt:

        ui.card(
            title="Ask AnomalyX",
            content=(
                "Investigate a product by describing your request "
                "in natural language."
            ),
            key="prompt_card"
        )

        user_prompt = st.text_area(
            "Investigation request",
            placeholder=(
                "Investigate product P015 on February 25, 2025"
            ),
            height=110
        )

        if ui.button("Investigate", key="run_prompt"):

            if not user_prompt.strip():
                st.warning("Please enter an investigation request.")

            else:
                try:
                    # Interpret natural-language request
                    parsed = interpret_prompt(user_prompt)

                    product_id = parsed.get("product_id")
                    date = parsed.get("date")

                    if not product_id or not date:
                        st.warning(
                            "Please include both a product ID and date."
                        )

                    else:
                        # Validate against available dataset
                        valid_record = features[
                            (features["product_id"] == product_id)
                            & (
                                features["date"]
                                == pd.to_datetime(date)
                            )
                        ]

                        if valid_record.empty:
                            st.warning(
                                "This product/date combination "
                                "does not exist in the dataset."
                            )

                        else:
                            with st.spinner(
                                f"Investigating {product_id} on {date}..."
                            ):
                                result = run_investigation(
                                    product_id,
                                    date
                                )

                            st.session_state["prompt_result"] = result
                            st.session_state["prompt_product"] = product_id
                            st.session_state["prompt_date"] = date

                except Exception as e:
                    st.error(f"Investigation failed: {e}")

        if "prompt_result" in st.session_state:

            display_investigation_report(
                result=st.session_state["prompt_result"],
                product_id=st.session_state["prompt_product"],
                date=st.session_state["prompt_date"]
            )



# =====================================================
# PAGE 3 — TECHNICAL DETAILS
# =====================================================
elif page == "Technical Details":

    st.markdown(
        '<div class="ax-section">Detection Model</div>',
        unsafe_allow_html=True
    )

    ui.card(
        title="Local Outlier Factor (LOF)",
        content=(
            "AnomalyX uses an unsupervised machine-learning "
            "model to detect unusual product behavior. "
            "Detected anomalies are ranked by their LOF scores."
        ),
        key="model_card"
    )

    st.write("")
    st.markdown("#### Ranked Detection Results")

    technical_df = anomalies[
        [
            "Anomaly Rank",
            "product_id",
            "date",
            "lof_score"
        ]
    ].copy()

    technical_df["date"] = (
        technical_df["date"].dt.strftime("%Y-%m-%d")
    )

    technical_df = technical_df.rename(columns={
        "product_id": "Product",
        "date": "Date",
        "lof_score": "LOF Score"
    })

    st.dataframe(
        technical_df,
        hide_index=True,
        width="stretch"
    )

    st.caption(
        "More negative LOF scores indicate stronger anomalies. "
        "The main dashboard uses Anomaly Rank instead."
    )
