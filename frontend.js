import React, { useState } from 'react';
import { LineChart, Line, BarChart, Bar, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Upload, UploadCloud } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

const TimeSeriesDashboard = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) throw new Error('Analysis failed');
      
      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderChart = (chartData, title, dataKey = 'value') => (
    <Card className="w-full h-[400px]">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="index" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey={dataKey} stroke="#8884d8" />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );

  return (
    <div className="container mx-auto p-4 bg-gray-50 min-h-screen">
      <h1 className="text-4xl font-bold mb-8 text-gray-800">Retail Sales Forecasting</h1>
      
      <div className="mb-8">
        <Card>
          <CardContent className="flex items-center justify-center p-6">
            <label className="flex flex-col items-center cursor-pointer">
              <UploadCloud className="w-12 h-12 text-gray-400 mb-2" />
              <span className="text-sm text-gray-500">Upload CSV file</span>
              <input type="file" className="hidden" accept=".csv" onChange={handleFileUpload} />
            </label>
          </CardContent>
        </Card>
      </div>

      {loading && (
        <Alert>
          <AlertTitle>Processing</AlertTitle>
          <AlertDescription>Analyzing your data...</AlertDescription>
        </Alert>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {data && (
        <Tabs defaultValue="eda" className="space-y-4">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="eda">EDA</TabsTrigger>
            <TabsTrigger value="stationarity">Stationarity</TabsTrigger>
            <TabsTrigger value="seasonality">Seasonality</TabsTrigger>
            <TabsTrigger value="forecast">Forecast</TabsTrigger>
          </TabsList>

          <TabsContent value="eda" className="space-y-4">
            {renderChart(data.missing_values.handled_data, 'Time Series Plot')}
            
            <div className="grid grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Descriptive Statistics</CardTitle>
                </CardHeader>
                <CardContent>
                  {/* Add descriptive statistics here */}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="stationarity" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Stationarity Test Results</CardTitle>
              </CardHeader>
              <CardContent>
                <p>ADF Statistic: {data.stationarity.adf_statistic}</p>
                <p>P-value: {data.stationarity.p_value}</p>
                <p>Is Stationary: {data.stationarity.is_stationary ? 'Yes' : 'No'}</p>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="seasonality" className="grid grid-cols-2 gap-4">
            {renderChart(data.seasonality.trend, 'Trend')}
            {renderChart(data.seasonality.seasonal, 'Seasonal')}
            {renderChart(data.seasonality.residual, 'Residual')}
          </TabsContent>

          <TabsContent value="forecast" className="space-y-4">
            {renderChart(data.forecast.forecast, 'Sales Forecast')}
            <Card>
              <CardHeader>
                <CardTitle>Model Information</CardTitle>
              </CardHeader>
              <CardContent>
                <p>Best Model: {data.forecast.model_info.model}</p>
                <p>AIC: {data.forecast.model_info.metrics.aic}</p>
                <p>BIC: {data.forecast.model_info.metrics.bic}</p>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
};

export default TimeSeriesDashboard;