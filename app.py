
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Intelligent Sales Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. COLORS
# =====================================================

NAVY = "#101827"
CARD = "#1B263B"
TEAL = "#00BFA6"
GOLD = "#F4B942"
TEXT = "#FFFFFF"
SECONDARY = "#8FA3B8"

# =====================================================
# 3. CUSTOM CSS
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #101827;
    color: #FFFFFF;
}

[data-testid="stSidebar"] {
    background-color: #1B263B;
    border-right: 1px solid #263B50;
}

[data-testid="stMetric"] {
    background-color: #1B263B;
    border: 1px solid #00BFA6;
    padding: 18px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #FFFFFF;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #00BFA6;
    font-weight: 800;
}

h1, h2, h3 {
    color: #FFFFFF;
    font-weight: 700;
}

.stButton > button {
    background-color: #00BFA6;
    color: #101827;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    min-height: 42px;
}

.stButton > button:hover {
    background-color: #009F8D;
    color: #FFFFFF;
}

.stDownloadButton > button {
    background-color: #1B263B;
    color: #00BFA6;
    border: 1px solid #00BFA6;
    border-radius: 8px;
}

hr {
    border-color: #00BFA6;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    sales = pd.read_csv("cleaned_sales_data.csv.gz", compression="gzip")
    forecast = pd.read_csv("future_forecast.csv")

    sales["InvoiceDate"] = pd.to_datetime(
        sales["InvoiceDate"]
    )

    forecast["InvoiceDate"] = pd.to_datetime(
        forecast["InvoiceDate"]
    )

    return sales, forecast


@st.cache_resource
def load_model():

    return joblib.load("best_sales_model.pkl")


try:

    sales, forecast = load_data()
    model = load_model()

except Exception as e:

    st.error(f"Error loading project files: {e}")

    st.info(
        "Make sure app.py, both CSV files and "
        "best_sales_model.pkl are in the same folder."
    )

    st.stop()


# =====================================================
# 5. HELPER FUNCTIONS
# =====================================================

def format_number(value):
    return f"{value:,.2f}"


def style_chart(fig):

    fig.update_layout(
        paper_bgcolor=NAVY,
        plot_bgcolor=CARD,
        font=dict(
            color=TEXT,
            family="Inter"
        ),
        title_font=dict(
            color=TEXT,
            size=18
        ),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=30
        )
    )

    return fig


def create_daily_features(history, next_date):

    features = pd.DataFrame({

        "Lag_1": [history.iloc[-1]],

        "Lag_7": [history.iloc[-7]],

        "Lag_14": [history.iloc[-14]],

        "DayOfWeek": [next_date.dayofweek],

        "Month": [next_date.month],

        "DayOfMonth": [next_date.day],

        "IsWeekend": [
            int(next_date.dayofweek in [5, 6])
        ],

        "Rolling_Mean_7": [
            history.iloc[-7:].mean()
        ],

        "Rolling_Std_7": [
            history.iloc[-7:].std()
        ]

    })

    return features


def generate_custom_forecast(history, number_of_days):

    forecast_history = history.copy()

    predictions = []

    for i in range(number_of_days):

        next_date = (
            forecast_history.index[-1]
            + pd.Timedelta(days=1)
        )

        next_features = create_daily_features(
            forecast_history,
            next_date
        )

        prediction = float(
            model.predict(next_features)[0]
        )

        # Saturday assumption used in the Notebook
        if next_date.dayofweek == 5:
            prediction = 0.0

        forecast_history.loc[next_date] = prediction

        predictions.append({
            "Date": next_date,
            "Predicted_Sales": prediction
        })

    return pd.DataFrame(predictions)


# =====================================================
# 6. SIDEBAR NAVIGATION & DEVELOPER BRANDING
# =====================================================

st.sidebar.title("📊 SALES INTELLIGENCE")

st.sidebar.caption(
    "Machine Learning Powered Analytics"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Executive Overview",
        "Sales Analytics",
        "Sales Forecast",
        "Custom Prediction",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("System Status: Online")

st.sidebar.markdown("---")

st.sidebar.caption("DEVELOPED BY")

st.sidebar.markdown(
    "<h3 style='color:#00BFA6;'>Wael Ayman</h3>",
    unsafe_allow_html=True
)

st.sidebar.caption("Data Science & AI")

st.sidebar.caption(
    "Intelligent Sales Forecasting System"
)


# =====================================================
# 7. EXECUTIVE OVERVIEW
# =====================================================

if page == "Executive Overview":

    st.title("INTELLIGENT SALES PERFORMANCE")

    st.caption(
        "A complete overview of business performance"
    )

    st.markdown("---")

    # KPIs

    total_sales = sales["TotalSales"].sum()

    total_orders = sales["InvoiceNo"].nunique()

    total_customers = sales["CustomerID"].nunique()

    total_quantity = sales["Quantity"].sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Sales",
        format_number(total_sales)
    )

    c2.metric(
        "Total Orders",
        f"{total_orders:,.0f}"
    )

    c3.metric(
        "Total Customers",
        f"{total_customers:,.0f}"
    )

    c4.metric(
        "Total Quantity",
        f"{total_quantity:,.0f}"
    )

    # Country Filter

    st.markdown("### Sales Filters")

    countries = sorted(
        sales["Country"].dropna().unique()
    )

    selected_country = st.selectbox(
        "Select Country",
        ["All Countries"] + countries,
        key="overview_country"
    )

    if selected_country == "All Countries":

        filtered_sales = sales.copy()

    else:

        filtered_sales = sales[
            sales["Country"] == selected_country
        ]

    # Daily Sales Trend

    st.markdown("### Daily Sales Trend")

    daily_sales = (
        filtered_sales.groupby(
            filtered_sales["InvoiceDate"].dt.date
        )["TotalSales"]
        .sum()
        .reset_index()
    )

    daily_sales.columns = ["Date", "Sales"]

    fig = px.line(
        daily_sales,
        x="Date",
        y="Sales",
        title="Daily Sales Performance",
        template="plotly_dark"
    )

    fig.update_traces(
        line_color=TEAL,
        line_width=2
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    # Country and Product Charts

    col1, col2 = st.columns(2)

    with col1:

        country_sales = (
            filtered_sales.groupby("Country")["TotalSales"]
            .sum()
            .nlargest(10)
            .reset_index()
        )

        fig_country = px.bar(
            country_sales,
            x="TotalSales",
            y="Country",
            orientation="h",
            title="Top 10 Countries",
            template="plotly_dark"
        )

        fig_country.update_traces(
            marker_color=TEAL
        )

        st.plotly_chart(
            style_chart(fig_country),
            use_container_width=True
        )

    with col2:

        product_sales = (
            filtered_sales.groupby("Description")["TotalSales"]
            .sum()
            .nlargest(5)
            .reset_index()
        )

        fig_products = px.pie(
            product_sales,
            names="Description",
            values="TotalSales",
            hole=0.6,
            title="Top 5 Products",
            template="plotly_dark"
        )

        st.plotly_chart(
            style_chart(fig_products),
            use_container_width=True
        )


# =====================================================
# 8. SALES ANALYTICS
# =====================================================

elif page == "Sales Analytics":

    st.title("SALES & RETURNS ANALYSIS")

    st.caption(
        "Explore sales performance and product returns"
    )

    st.markdown("---")

    # KPIs

    total_sales = sales["TotalSales"].sum()

    total_orders = sales["InvoiceNo"].nunique()

    total_returns = abs(
        sales.loc[
            sales["Quantity"] < 0,
            "Quantity"
        ].sum()
    )

    total_products = sales["StockCode"].nunique()

    average_order = (
        total_sales / total_orders
        if total_orders > 0 else 0
    )

    gross_quantity = (
        sales["Quantity"].sum() + total_returns
    )

    return_rate = (
        total_returns / gross_quantity * 100
        if gross_quantity > 0 else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Average Order Value",
        format_number(average_order)
    )

    c2.metric(
        "Returned Quantity",
        f"{total_returns:,.0f}"
    )

    c3.metric(
        "Total Products",
        f"{total_products:,.0f}"
    )

    c4.metric(
        "Return Rate",
        f"{return_rate:.2f}%"
    )

    # Month Filter

    st.markdown("### Monthly Analysis")

    available_months = sorted(
        sales["InvoiceDate"].dt.strftime("%Y-%m").unique()
    )

    selected_month = st.selectbox(
        "Select Month",
        ["All Months"] + available_months,
        key="analytics_month"
    )

    if selected_month == "All Months":

        monthly_data = sales.copy()

    else:

        monthly_data = sales[
            sales["InvoiceDate"].dt.strftime("%Y-%m")
            == selected_month
        ].copy()

    # Monthly Sales Chart

    monthly_sales = (
        monthly_data.groupby(
            monthly_data["InvoiceDate"].dt.to_period("M")
        )["TotalSales"]
        .sum()
        .reset_index()
    )

    monthly_sales["Month"] = (
        monthly_sales["InvoiceDate"].astype(str)
    )

    fig_month = px.bar(
        monthly_sales,
        x="Month",
        y="TotalSales",
        title="Monthly Sales Performance",
        template="plotly_dark"
    )

    fig_month.update_traces(
        marker_color=TEAL
    )

    st.plotly_chart(
        style_chart(fig_month),
        use_container_width=True
    )

    # Returns by Country

    st.markdown("### Returns Analysis")

    returns = monthly_data[
        monthly_data["Quantity"] < 0
    ].copy()

    returns["ReturnsSales"] = -returns["TotalSales"]

    country_returns = (
        returns.groupby("Country")["ReturnsSales"]
        .sum()
        .nlargest(10)
        .reset_index()
    )

    if not country_returns.empty:

        fig_returns = px.pie(
            country_returns,
            names="Country",
            values="ReturnsSales",
            hole=0.6,
            title="Returns Sales by Country",
            template="plotly_dark"
        )

        st.plotly_chart(
            style_chart(fig_returns),
            use_container_width=True
        )

    else:

        st.info("No return records for the selected month.")


# =====================================================
# 9. SALES FORECAST
# =====================================================

elif page == "Sales Forecast":

    st.title("SALES FORECASTING")

    st.caption(
        "30-Day Sales Forecast powered by Neural Network"
    )

    st.markdown("---")

    forecast_total = forecast[
        "Predicted_Sales"
    ].sum()

    forecast_average = forecast[
        "Predicted_Sales"
    ].mean()

    forecast_max = forecast[
        "Predicted_Sales"
    ].max()

    positive_forecasts = forecast.loc[
        forecast["Predicted_Sales"] > 0,
        "Predicted_Sales"
    ]

    forecast_min = (
        positive_forecasts.min()
        if not positive_forecasts.empty else 0
    )

    # KPIs

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Forecast Total Sales",
        format_number(forecast_total)
    )

    c2.metric(
        "Average Daily Sales",
        format_number(forecast_average)
    )

    c3.metric(
        "Maximum Forecast",
        format_number(forecast_max)
    )

    c4.metric(
        "Minimum Forecast",
        format_number(forecast_min)
    )

    # Daily Forecast

    st.markdown("### Daily Sales Forecast")

    fig_forecast = px.line(
        forecast,
        x="InvoiceDate",
        y="Predicted_Sales",
        markers=True,
        title="Next 30 Days Sales Prediction",
        template="plotly_dark"
    )

    fig_forecast.update_traces(
        line_color=TEAL,
        marker_color=GOLD
    )

    st.plotly_chart(
        style_chart(fig_forecast),
        use_container_width=True
    )

    # Monthly Forecast

    st.markdown("### Monthly Forecast")

    monthly_forecast = (
        forecast.assign(
            Month=forecast["InvoiceDate"].dt.to_period("M").astype(str)
        )
        .groupby("Month")["Predicted_Sales"]
        .sum()
        .reset_index()
    )

    fig_monthly = px.bar(
        monthly_forecast,
        x="Month",
        y="Predicted_Sales",
        title="Forecast Sales by Month",
        template="plotly_dark"
    )

    fig_monthly.update_traces(
        marker_color=TEAL
    )

    st.plotly_chart(
        style_chart(fig_monthly),
        use_container_width=True
    )

    # Forecast Details

    st.markdown("### Forecast Details")

    st.dataframe(
        forecast,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Forecast CSV",
        data=forecast.to_csv(index=False).encode("utf-8"),
        file_name="sales_forecast.csv",
        mime="text/csv"
    )

    st.info(
        "Forecasts are estimates based on historical sales patterns. "
        "Saturday sales are assumed to be zero based on the observed dataset."
    )


# =====================================================
# 10. CUSTOM PREDICTION
# =====================================================

elif page == "Custom Prediction":

    st.title("CUSTOM SALES PREDICTION")

    st.caption(
        "Generate your own sales forecast using the trained Neural Network"
    )

    st.markdown("---")

    st.markdown("### Configure Your Forecast")

    col1, col2 = st.columns([1, 2])

    with col1:

        forecast_days = st.selectbox(
            "Select Forecast Period",
            [7, 14, 30],
            index=2
        )

        run_prediction = st.button(
            "Run Prediction",
            use_container_width=True
        )

    with col2:

        st.info(
            "Choose your forecast period and run the trained "
            "Neural Network to generate new daily predictions."
        )

    if run_prediction:

        daily_history = (
            sales.set_index("InvoiceDate")["TotalSales"]
            .resample("D")
            .sum()
        )

        if len(daily_history) < 14:

            st.error(
                "Not enough historical data to generate predictions."
            )

        else:

            with st.spinner("Generating sales predictions..."):

                custom_forecast = generate_custom_forecast(
                    daily_history,
                    forecast_days
                )

            st.success(
                "Prediction completed successfully!"
            )

            # Prediction KPIs

            total_prediction = custom_forecast[
                "Predicted_Sales"
            ].sum()

            average_prediction = custom_forecast[
                "Predicted_Sales"
            ].mean()

            maximum_prediction = custom_forecast[
                "Predicted_Sales"
            ].max()

            st.markdown("### Prediction Results")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Total Predicted Sales",
                format_number(total_prediction)
            )

            c2.metric(
                "Average Daily Sales",
                format_number(average_prediction)
            )

            c3.metric(
                "Maximum Daily Sales",
                format_number(maximum_prediction)
            )

            # Prediction Chart

            st.markdown("### Forecast Visualization")

            fig_custom = px.line(
                custom_forecast,
                x="Date",
                y="Predicted_Sales",
                markers=True,
                title=f"{forecast_days}-Day Sales Forecast",
                template="plotly_dark"
            )

            fig_custom.update_traces(
                line_color=TEAL,
                marker_color=GOLD,
                line_width=3
            )

            st.plotly_chart(
                style_chart(fig_custom),
                use_container_width=True
            )

            # Results Table

            st.markdown("### Daily Prediction Details")

            st.dataframe(
                custom_forecast,
                use_container_width=True,
                hide_index=True
            )

            # Download

            st.download_button(
                "Download Prediction Results",
                data=custom_forecast.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="custom_sales_prediction.csv",
                mime="text/csv"
            )

            st.info(
                "These predictions are generated recursively from "
                "historical daily sales. Saturday sales are assumed "
                "to be zero, following the Notebook assumption."
            )

    else:

        st.markdown("---")

        st.markdown("### Ready to Predict?")

        st.write(
            "Select 7, 14, or 30 days and click "
            "Run Prediction to generate your forecast."
        )


# =====================================================
# 11. ABOUT PROJECT
# =====================================================

elif page == "About Project":

    st.title("ABOUT THE PROJECT")

    st.caption(
        "Project information, methodology and technologies"
    )

    st.markdown("---")

    st.markdown("""
    ## Intelligent Sales Forecasting System

    An End-to-End Data Science project designed to analyze
    historical retail sales and forecast future sales.

    ### Project Developer

    **Wael Ayman**

    Data Science & Artificial Intelligence

    ### Project Workflow

    1. Data Collection
    2. Data Cleaning and Preprocessing
    3. Exploratory Data Analysis (EDA)
    4. Feature Engineering
    5. Machine Learning Model Training
    6. Neural Network Forecasting
    7. Power BI Dashboard
    8. Streamlit Web Application

    ### Machine Learning Models

    - Linear Regression
    - Random Forest Regressor
    - Neural Network (MLPRegressor)
    - XGBoost Regressor

    ### Technology Stack

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - XGBoost
    - Plotly
    - Streamlit
    - Power BI

    ### Forecasting Methodology

    The selected Neural Network model uses historical sales
    features such as lag values and rolling statistics.

    The application supports recursive daily forecasts
    for 7, 14, or 30 days.

    ### Important Assumption

    Saturday sales are assumed to be zero based on the
    observed historical dataset.

    Forecast values are estimates and should be interpreted
    with caution.
    """)

    st.markdown("---")

    st.success(
        "Intelligent Sales Forecasting System"
    )

    st.markdown(
        "<h3 style='color:#00BFA6;'>Developed by Wael Ayman</h3>",
        unsafe_allow_html=True
    )

    st.caption(
        "Data Science & Artificial Intelligence"
    )
