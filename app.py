import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import io
import statsmodels.api as sm  


st.set_page_config(page_title="Retail Sales Forecasting", layout="wide", initial_sidebar_state="expanded")

#Custom CSS for UI
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
        color: #fafafa;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    .Widget>label {
        color: #fafafa;
        font-family: monospace;
    }
    .stButton>button {
        color: #4f8bf9;
        background-color: #0e1117;
        border: 1px solid #4f8bf9;
    }
    .stTextInput>div>div>input {
        color: #4f8bf9;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Retail Sales Forecasting")

#1. Data Collection
st.header("📁 Data Upload")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m')
    data.set_index('Date', inplace=True)
    data = data.asfreq('MS')  # Set the frequency to month start

    #2. Exploratory Data Analysis
    st.header("📊 Exploratory Data Analysis")

    #3. Visualizations
    st.subheader("📈 Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Line Graph")
        fig, ax = plt.subplots()
        ax.plot(data.index, data['Sales'])
        ax.set_title("Sales Over Time")
        st.pyplot(fig)

        st.write("Histogram")
        fig, ax = plt.subplots()
        ax.hist(data['Sales'], bins=30)
        ax.set_title("Sales Distribution")
        st.pyplot(fig)

    with col2:
        st.write("Scatter Plot")
        fig, ax = plt.subplots()
        ax.scatter(data.index, data['Sales'])
        ax.set_title("Sales Scatter Plot")
        st.pyplot(fig)

        st.write("QQ Plot")
        fig, ax = plt.subplots()
        sm.qqplot(data['Sales'], ax=ax)
        ax.set_title("QQ Plot")
        st.pyplot(fig)

    #4. Descriptive Statistics
    st.subheader("📉 Descriptive Statistics")
    st.write(data.describe())

    #5. Missing Values
    st.subheader("🕵️ Missing Values")
    missing_values = data.isnull().sum()
    st.write(missing_values)

    if missing_values.sum() > 0:
        st.write("Handling missing values...")

    else:
        st.write("No missing values found.")

    #6. Stationarity Check
    st.subheader("📊 Stationarity Check")

    def adf_test(series):
        result = adfuller(series)
        st.write(f'ADF Statistic: {result[0]}')
        st.write(f'p-value: {result[1]}')
        st.write('Critical Values:')
        for key, value in result[4].items():
            st.write(f'\t{key}: {value}')
        
        # Interpretation
        with st.expander("Interpreting ADF Test Results"):
            st.write("""
            The Augmented Dickey-Fuller (ADF) test is used to determine if a time series is stationary.

            **Null Hypothesis (H0)**: The time series has a unit root (non-stationary)
            **Alternative Hypothesis (H1)**: The time series does not have a unit root (stationary)

            **Interpretation**:
            1. If p-value <= 0.05: Reject the null hypothesis. The time series is stationary.
            2. If p-value > 0.05: Fail to reject the null hypothesis. The time series is non-stationary.

            **Confidence Intervals**:
            - The critical values represent different confidence levels (1%, 5%, 10%).
            - If the ADF statistic is more negative than the critical value, we reject the null hypothesis at that confidence level.

            **Example**:
            If ADF statistic < Critical Value at 5%, we can say with 95% confidence that the time series is stationary.
            """)

    adf_test(data['Sales'])

    #7. Seasonality Decomposition
    st.subheader("🌊 Seasonality Decomposition")

    decomposition = seasonal_decompose(data['Sales'], model='additive', period=12)
    fig = decomposition.plot()
    st.pyplot(fig)

    with st.expander("What is Seasonality Decomposition?"):
        st.write("""
        Seasonality decomposition involves breaking down a time series into its components: trend, seasonality, and residuals.
        - **Trend**: The long-term movement in the data.
        - **Seasonality**: The repeating short-term cycle in the data.
        - **Residuals**: The random noise in the data.
        """)

    #8. ACF and PACF plots
    st.subheader("📊 ACF and PACF Plots")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(data['Sales'], ax=ax1)
    plot_pacf(data['Sales'], ax=ax2)
    st.pyplot(fig)

    with st.expander("What are ACF and PACF?"):
        st.write("""
        - **ACF (Autocorrelation Function)**: Measures the correlation between the time series and its lagged values.
        - **PACF (Partial Autocorrelation Function)**: Measures the correlation between the time series and its lagged values, controlling for the values of the time series at all shorter lags.
        """)

    #9. Model Selection
    st.subheader("🤖 Model Selection")

    #Auto ARIMA
    auto_model = auto_arima(data['Sales'], seasonal=True, m=12)
    st.write("Best ARIMA model:", auto_model.order)
    st.write("Best seasonal order:", auto_model.seasonal_order)

    with st.expander("Understanding ARIMA and SARIMA Models"):
        st.write("""
        - **ARIMA (AutoRegressive Integrated Moving Average)**: A model used for forecasting time series data.
          - **Order (p, d, q)**: 
            - **p**: Number of lag observations included in the model (lag order).
            - **d**: Number of times that the raw observations are differenced (degree of differencing).
            - **q**: Size of the moving average window.
        - **SARIMA (Seasonal ARIMA)**: An extension of ARIMA that supports univariate time series data with a seasonal component.
          - **Seasonal Order (P, D, Q, m)**:
            - **P**: Seasonal autoregressive order.
            - **D**: Seasonal differencing order.
            - **Q**: Seasonal moving average order.
            - **m**: Number of time steps for a single seasonal period.
        """)

    with st.expander("Interpreting ARIMA Results"):
        st.write("""
        When interpreting ARIMA results:

        1. **Coefficient Significance**: Look at the p-values for each coefficient. If p-value < 0.05, the coefficient is statistically significant.

        2. **Residual Analysis**: 
           - Residuals should be normally distributed with mean zero.
           - There should be no autocorrelation in residuals.

        3. **Information Criteria**: Lower AIC (Akaike Information Criterion) or BIC (Bayesian Information Criterion) values indicate better models.

        4. **Forecast Confidence Intervals**: 
           - Narrower intervals indicate more precise predictions.
           - 95% CI means we're 95% confident the true value will fall within this range.

        5. **Model Comparison**: Compare different ARIMA models using these metrics to choose the best one.

        Remember, a good statistical fit doesn't always guarantee good forecasts. Always validate your model with out-of-sample data.
        """)

    #10. Model Deployment
    st.subheader("🚀 Model Deployment")

    n_periods = st.slider("Select number of periods to forecast", 1, 24, 12)

    model = ARIMA(data['Sales'], order=auto_model.order, seasonal_order=auto_model.seasonal_order)
    results = model.fit()

    # Display model summary
    with st.expander("Model Summary"):
        buf = io.StringIO()
        results.summary().tables[1].to_csv(buf)
        st.write(buf.getvalue())

    #Generating future dates
    future_dates = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(1), periods=n_periods, freq='MS')

    #Forecast
    forecast = results.get_forecast(steps=n_periods)
    mean_forecast = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    #Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Sales'], label='Historical Data')
    ax.plot(future_dates, mean_forecast, label='Forecast', color='red')
    ax.fill_between(future_dates, 
                    confidence_intervals.iloc[:, 0], 
                    confidence_intervals.iloc[:, 1], 
                    color='pink', alpha=0.3, label='95% Confidence Interval')
    ax.set_title("Sales Forecast with Confidence Intervals")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.write("Forecast Values:")
    forecast_df = pd.DataFrame({'Forecast': mean_forecast, 
                                'Lower CI': confidence_intervals.iloc[:, 0],
                                'Upper CI': confidence_intervals.iloc[:, 1]}, 
                               index=future_dates)
    st.write(forecast_df)

else:
    st.info("Please upload a CSV file to begin the analysis.")
