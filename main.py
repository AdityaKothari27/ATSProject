import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TimeSeriesAnalyzer:
    def __init__(self, data):
        self.data = pd.Series(data)
        self.original_data = self.data.copy()
    
    def handle_missing_values(self):
        return {
            "original_missing": self.data.isnull().sum(),
            "handled_data": self.data.interpolate(method='linear').tolist()
        }
    
    def check_stationarity(self):
        result = adfuller(self.data.dropna())
        return {
            "adf_statistic": result[0],
            "p_value": result[1],
            "is_stationary": result[1] < 0.05
        }
    
    def apply_transformation(self, method='log'):
        if method == 'log':
            self.data = np.log1p(self.data)
        elif method == 'diff':
            self.data = self.data.diff().dropna()
        return self.data.tolist()
    
    def decompose_seasonality(self):
        decomposition = seasonal_decompose(self.data, period=12)
        return {
            "trend": decomposition.trend.dropna().tolist(),
            "seasonal": decomposition.seasonal.dropna().tolist(),
            "residual": decomposition.resid.dropna().tolist()
        }
    
    def calculate_acf_pacf(self, nlags=40):
        acf_values = pd.Series(plot_acf(self.data, lags=nlags, alpha=0.05))
        pacf_values = pd.Series(plot_pacf(self.data, lags=nlags, alpha=0.05))
        return {
            "acf": acf_values.tolist(),
            "pacf": pacf_values.tolist()
        }
    
    def fit_models(self):
        results = {}
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(self.data, order=(p, d, q))
                        result = model.fit()
                        results[f'ARIMA({p},{d},{q})'] = {
                            'aic': result.aic,
                            'bic': result.bic
                        }
                    except:
                        continue
        return results
    
    def get_best_model(self):
        results = self.fit_models()
        best_model = min(results.items(), key=lambda x: x[1]['aic'])
        return {
            'model': best_model[0],
            'metrics': best_model[1]
        }
    
    def forecast(self, steps=30):
        best_model_info = self.get_best_model()
        p, d, q = map(int, best_model_info['model'].strip('ARIMA()').split(','))
        model = ARIMA(self.data, order=(p, d, q))
        result = model.fit()
        forecast = result.forecast(steps=steps)
        return {
            'forecast': forecast.tolist(),
            'model_info': best_model_info
        }

@app.post("/analyze")
async def analyze_data(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(pd.io.common.BytesIO(contents))
    
    # Assuming the file has 'date' and 'sales' columns
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    analyzer = TimeSeriesAnalyzer(df['sales'])
    
    missing_values = analyzer.handle_missing_values()
    stationarity = analyzer.check_stationarity()
    seasonality = analyzer.decompose_seasonality()
    acf_pacf = analyzer.calculate_acf_pacf()
    forecast_result = analyzer.forecast()
    
    return {
        "missing_values": missing_values,
        "stationarity": stationarity,
        "seasonality": seasonality,
        "acf_pacf": acf_pacf,
        "forecast": forecast_result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)