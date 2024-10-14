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

    adf_test(data['Sales'])

    #7. Seasonality Decomposition
    st.subheader("🌊 Seasonality Decomposition")

    decomposition = seasonal_decompose(data['Sales'], model='additive', period=12)
    fig = decomposition.plot()
    st.pyplot(fig)

    #8. ACF and PACF plots
    st.subheader("📊 ACF and PACF Plots")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(data['Sales'], ax=ax1)
    plot_pacf(data['Sales'], ax=ax2)
    st.pyplot(fig)

    #9. Model Selection
    st.subheader("🤖 Model Selection")

    #Auto ARIMA
    auto_model = auto_arima(data['Sales'], seasonal=True, m=12)
    st.write("Best ARIMA model:", auto_model.order)
    st.write("Best seasonal order:", auto_model.seasonal_order)

    #10. Model Deployment
    st.subheader("🚀 Model Deployment")

    n_periods = st.slider("Select number of periods to forecast", 1, 24, 12)

    model = ARIMA(data['Sales'], order=auto_model.order, seasonal_order=auto_model.seasonal_order)
    results = model.fit()

    #Generating future dates
    future_dates = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(1), periods=n_periods, freq='MS')

    #Forecast
    forecast = results.forecast(steps=n_periods)
    forecast_df = pd.DataFrame({'Forecast': forecast}, index=future_dates)

    #Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Sales'], label='Historical Data')
    ax.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red')
    ax.set_title("Sales Forecast")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.write("Forecast Values:")
    st.write(forecast_df)

else:
    st.info("Please upload a CSV file to begin the analysis.")
