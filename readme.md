**Project Overview**

This project develops a high-precision hourly power demand forecasting system for the Bangladesh power grid. By combining historical grid data, real-time weather metrics, and long-term economic indicators, we aim to provide reliable predictions to help grid operators manage load and minimize load shedding.

**Datasets**

\* PGCB Power Data: Hourly records of generation and demand (2015-2025).

\* Open-Meteo Weather: Meteorological data for Dhaka (Temperature, Humidity, etc.).

\* World Bank Economic Data: Macroeconomic indicators (GDP, Urbanization).

**Key Features**

\* Pre-processing: Automated outlier detection (1.5x IQR) and KNN-based imputation for missing renewable data.

\* Engineering: Cyclical temporal features, multi-step lags (1h, 24h, 168h), and rolling aggregate statistics.

\* Model: A weighted ensemble of Random Forest, XGBoost, and LightGBM.

**Performance**

\* Best Model: Weighted Ensemble

\* MAPE: 3.15%

\* MAE: 323 MW

**Usage**

1\. Run the Setup cell and upload the three required CSV files.

2\. Execute the notebook sequentially to perform EDA, Preprocessing, and Model Training.

3\. The final section generates visualizations comparing actual vs. predicted demand.





