# 📈 Reliance Industries Stock Price Forecasting

## 🔗 Project Links

🌐 **Live Streamlit Application:**
https://reliance-industries-stock-price-forecasting-zglv7eoqyzfaxv2eid.streamlit.app/


📓 **Jupyter Notebook:**
`Reliance_Stock_Forecasting.ipynb`

## 📌 Project Overview

This project focuses on **Reliance Industries stock price forecasting** using **Time Series Forecasting and Long Short-Term Memory (LSTM)** neural networks.

The objective is to analyze historical stock price data, identify short-term and long-term trends, evaluate multiple forecasting models, select the best-performing model based on forecasting errors, and predict the **next 30 trading days** of Reliance Industries' closing price.

The complete project follows an end-to-end workflow:

**Data Preprocessing → Exploratory Data Analysis → Train-Test Split → Model Building → Model Evaluation → Model Selection → Final Forecasting → Streamlit Deployment**

---

## 🎯 Business Objective

The main objective of this project is to forecast the **closing price of Reliance Industries for the next 30 trading days** using historical stock-price data.

The project aims to:

* Analyze historical stock-price behavior
* Understand short-term and long-term trends
* Analyze daily returns and volatility
* Compare different time-series forecasting techniques
* Evaluate models using MAE, RMSE and MAPE
* Select the best-performing forecasting model
* Generate a 30-trading-day forecast
* Deploy the final forecasting model using Streamlit

---

## 📊 Dataset

The dataset contains historical Reliance Industries stock-market information.

### Dataset Features

| Column      | Description                          |
| ----------- | ------------------------------------ |
| `Date`      | Trading date                         |
| `Open`      | Opening price                        |
| `High`      | Highest price during the trading day |
| `Low`       | Lowest price during the trading day  |
| `Close`     | Closing price                        |
| `Adj Close` | Adjusted closing price               |
| `Volume`    | Trading volume                       |

### Dataset Size

* **Rows:** 753
* **Columns:** 7
* **Target Variable:** `Close`

---

## 🔄 Project Workflow

```text
Historical Stock Data
        │
        ▼
Data Loading
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Exploratory Data Analysis
        │
        ├── Closing Price Trend
        ├── Daily Returns
        ├── Volatility
        ├── MA20
        ├── MA200
        └── Correlation Analysis
        │
        ▼
Train-Test Split
        │
        ▼
Model Building
        │
        ├── Naive
        ├── ARIMA
        ├── SARIMA
        ├── Exponential Smoothing
        └── LSTM
        │
        ▼
Model Evaluation
        │
        ├── MAE
        ├── RMSE
        └── MAPE
        │
        ▼
Best Model Selection
        │
        ▼
Final LSTM Retraining
        │
        ▼
30 Trading-Day Forecast
        │
        ▼
Streamlit Deployment
```

---

# 🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the behavior of Reliance Industries' stock price.

### Analysis Performed

### 1. Closing Price Trend

The historical closing price was visualized to understand overall price movements and trends over time.

### 2. Daily Returns

Daily returns were calculated to understand day-to-day percentage changes in the stock price.

### 3. Volatility Analysis

Daily returns were analyzed to understand the variability and volatility of the stock.

### 4. Moving Averages

Two moving averages were used:

* **MA20** — Short-term trend
* **MA200** — Long-term trend

These moving averages help identify changes in the short-term and long-term direction of the stock price.

### 5. Correlation Analysis

A correlation analysis was performed to understand relationships between the numerical variables.

---

# 🤖 Models Implemented

Five forecasting approaches were evaluated.

## 1. Naive Forecasting

The Naive model was used as a baseline forecasting approach.

It provides a simple benchmark against which more advanced models can be compared.

---

## 2. ARIMA

**ARIMA — AutoRegressive Integrated Moving Average**

ARIMA was implemented to model the time-series behavior of the closing price.

### Configuration

```text
ARIMA(5,1,0)
```

---

## 3. SARIMA

**SARIMA — Seasonal AutoRegressive Integrated Moving Average**

SARIMA was evaluated to determine whether incorporating seasonal behavior would improve forecasting performance.

### Configuration

```text
SARIMA(1,1,1) × (1,1,1,5)
```

---

## 4. Exponential Smoothing

Exponential Smoothing was implemented as another statistical forecasting approach for modeling the underlying behavior of the time series.

---

## 5. LSTM

**LSTM — Long Short-Term Memory**

LSTM is a recurrent neural-network architecture designed to learn patterns from sequential data.

For this project, the closing price was scaled using **MinMaxScaler** before training.

### LSTM Configuration

| Parameter         | Value              |
| ----------------- | ------------------ |
| Lookback Window   | 60 days            |
| First LSTM Layer  | 64 units           |
| Second LSTM Layer | 32 units           |
| Dropout           | Used               |
| Optimizer         | Adam               |
| Loss Function     | Mean Squared Error |
| Epochs            | 30                 |
| Batch Size        | 32                 |

The model uses the previous **60 observations** to predict the next closing price.

---

# 📏 Model Evaluation

The forecasting models were evaluated using:

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

### RMSE — Root Mean Squared Error

Penalizes larger prediction errors more strongly.

### MAPE — Mean Absolute Percentage Error

Measures prediction error as a percentage of the actual value.

---

# 🏆 Model Comparison

| Model                 |       MAE |      RMSE |      MAPE |
| --------------------- | --------: | --------: | --------: |
| **LSTM**              | **20.34** | **24.43** | **5.72%** |
| Naive                 |    123.63 |    135.62 |    33.08% |
| ARIMA                 |    123.93 |    135.90 |    33.17% |
| Exponential Smoothing |    198.93 |    220.29 |    53.11% |
| SARIMA                |    217.42 |    241.08 |    58.05% |

## 🥇 Best Model: LSTM

Based on MAE, RMSE and MAPE, **LSTM achieved the lowest forecasting errors** among the evaluated models.

Therefore, LSTM was selected as the final forecasting model.

---

# 🔁 Why LSTM Was Trained Twice

The LSTM was used in two stages for different purposes.

### Stage 1 — Model Evaluation

The first LSTM was trained using the training dataset and evaluated on the last-year test dataset.

The purpose was to compare LSTM fairly with:

* Naive
* ARIMA
* SARIMA
* Exponential Smoothing

### Stage 2 — Final Forecasting

After LSTM was identified as the best-performing model, the final LSTM was retrained using the **complete historical dataset**.

This allows the final model to learn from all available historical observations before generating future forecasts.

---

# 🔮 30-Trading-Day Forecast

After retraining the final LSTM, a recursive forecasting approach was used.

The forecasting process works as follows:

```text
Last 60 Historical Values
          │
          ▼
     LSTM Prediction
          │
          ▼
Add Prediction to Sequence
          │
          ▼
Predict Next Day
          │
          ▼
Repeat
          │
          ▼
30 Trading-Day Forecast
```

The final forecast contains:

* Future trading dates
* Predicted closing prices

The forecast was also visualized to make the predicted price movement easier to understand.

---

# 🚀 Streamlit Deployment

The final LSTM model was deployed using **Streamlit**.

The application provides an interactive interface for stock-price forecasting.

### Application Features

* Upload historical stock data
* Support for CSV/XLSX input
* Validate date and closing-price columns
* Apply the saved scaler
* Load the trained LSTM model
* Generate 30-trading-day forecasts
* Display forecast results
* Visualize predicted prices

---

# 🖥️ Application Workflow

```text
User Uploads CSV / Excel
          │
          ▼
Data Validation
          │
          ▼
Date & Close Column Detection
          │
          ▼
Data Preprocessing
          │
          ▼
Load Saved Scaler
          │
          ▼
Load Trained LSTM
          │
          ▼
Generate 30-Day Forecast
          │
          ▼
Display Results & Visualization
```

---

# 💾 Saved Model Files

The following trained artifacts are used for deployment:

```text
reliance_lstm_model.keras
reliance_stock_scaler.pkl
Reliance_LSTM_30_Day_Forecast.csv
```

### `reliance_lstm_model.keras`

Contains the trained final LSTM forecasting model.

### `reliance_stock_scaler.pkl`

Contains the scaler used to transform the stock-price data.

### `Reliance_LSTM_30_Day_Forecast.csv`

Contains the generated 30-trading-day forecast results.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Statistical Forecasting

* Statsmodels

### Deep Learning

* TensorFlow
* Keras

### Machine Learning

* Scikit-learn

### Deployment

* Streamlit

### Model Persistence

* Joblib
* Keras model format

---

# 📦 Installation

Clone the repository:

```bash
git clone [https://github.com/Rajeswari016/Reliance-Industries-Stock-Price-Forecasting].git
```

Move into the project directory:

```bash
cd Reliance_Stock_Forecasting
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Jupyter Notebook

Open the notebook:

```text
Reliance_Stock_Forecasting.ipynb
```

Run the cells sequentially to reproduce the analysis, model training, evaluation and forecasting workflow.

---

# 🌐 Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📁 Project Structure

```text
Reliance_Stock_Forecasting/
│
├── 📓 Reliance_Stock_Forecasting.ipynb
│
├── 📄 Reliance Industries Stock Forecasting.pdf
│
├── 🌐 app.py
│
├── 🧠 reliance_lstm_model.keras
│
├── 📊 reliance_stock_scaler.pkl
│
├── 📈 Reliance_LSTM_30_Day_Forecast.csv
│
├── 📦 requirements.txt

```

---

# 📌 Key Findings

* Historical stock-price behavior was analyzed through EDA.
* Daily returns and volatility were examined.
* MA20 and MA200 were used to understand short-term and long-term trends.
* Five forecasting approaches were evaluated.
* LSTM achieved the lowest MAE, RMSE and MAPE.
* LSTM was selected as the final forecasting model.
* The final LSTM was retrained using the complete historical dataset.
* A recursive forecasting approach was used to predict the next 30 trading days.
* The final model was deployed using Streamlit.

---

# 🎓 Project Outcome

This project demonstrates an end-to-end **time-series forecasting workflow**, from raw historical stock data to a deployed forecasting application.

The project combines:

**Data Analysis + Time Series Forecasting + Deep Learning + Model Evaluation + Deployment**

It demonstrates practical implementation of statistical forecasting methods and LSTM-based deep learning for financial time-series analysis.

---

# ⚠️ Disclaimer

This project is intended **for educational and research purposes only**.

The predictions generated by this application should not be considered financial advice or a recommendation to buy or sell any security.

Actual stock prices can be affected by market conditions, economic events, company announcements, investor sentiment and many other factors that may not be captured by the model.

---

# 👨‍💻 Developed By

## Dama Rajeswari

📧 Email: [damarajeswaridama@gmail.com]

🔗 GitHub:
https://github.com/Rajeswari016/Reliance-Industries-Stock-Price-Forecasting

---

