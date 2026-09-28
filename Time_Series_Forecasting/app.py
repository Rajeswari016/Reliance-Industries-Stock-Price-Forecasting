import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Reliance Stock Forecasting",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "reliance_lstm_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "reliance_stock_scaler.pkl")

SEQUENCE_LENGTH = 60
FORECAST_DAYS = 30


# =========================================================
# LOAD MODEL AND SCALER
# =========================================================

@st.cache_resource
def load_model_and_scaler():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    return model, scaler


try:

    model, scaler = load_model_and_scaler()

except FileNotFoundError:

    st.error(
        "❌ Model or scaler file not found."
    )

    st.info(
        "Keep the model and scaler in the same folder as app.py."
    )

    st.code(
        "reliance_lstm_model.keras\n"
        "reliance_stock_scaler.pkl"
    )

    st.stop()

except Exception as e:

    st.error(
        "❌ Unable to load the LSTM model or scaler."
    )

    st.exception(e)

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.title(
    "📈 Reliance Stock Price Forecasting"
)

st.subheader(
    "LSTM-Based 30 Trading-Day Forecast"
)

st.write(
    "Forecast Reliance stock closing prices using "
    "the final trained LSTM model."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "📊 Forecast Settings"
)

st.sidebar.info(
    f"""
**Model:** LSTM

**Forecast Horizon:** {FORECAST_DAYS} Trading Days

**Sequence Length:** {SEQUENCE_LENGTH}

**Target:** Close Price
"""
)

st.sidebar.header(
    "📂 Upload Stock Data"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Reliance Stock Data",
    type=["csv", "xlsx"]
)


# =========================================================
# FILE CHECK
# =========================================================

if uploaded_file is None:

    st.info(
        "👈 Please upload your Reliance stock data."
    )

    st.stop()


# =========================================================
# READ CSV / EXCEL
# =========================================================

try:

    if uploaded_file.name.lower().endswith(".csv"):

        data = pd.read_csv(
            uploaded_file
        )

    else:

        data = pd.read_excel(
            uploaded_file
        )

except Exception as e:

    st.error(
        "❌ Unable to read the uploaded file."
    )

    st.exception(e)

    st.stop()


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

data.columns = (
    data.columns
    .str.strip()
)


# =========================================================
# FIND CLOSE COLUMN
# =========================================================

close_column = None

for column in [
    "Close",
    "close",
    "CLOSE",
    "Adj Close",
    "adj_close"
]:

    if column in data.columns:

        close_column = column

        break


if close_column is None:

    st.error(
        "❌ Close price column not found."
    )

    st.write(
        "Available columns:"
    )

    st.write(
        list(data.columns)
    )

    st.stop()


# =========================================================
# FIND DATE COLUMN
# =========================================================

date_column = None

for column in [
    "Date",
    "date",
    "DATE"
]:

    if column in data.columns:

        date_column = column

        break


# =========================================================
# PROCESS DATE
# =========================================================

if date_column is not None:

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    data = data.dropna(
        subset=[date_column]
    )

    data = data.sort_values(
        date_column
    )


# =========================================================
# PROCESS CLOSE PRICE
# =========================================================

data[close_column] = pd.to_numeric(
    data[close_column],
    errors="coerce"
)

data = data.dropna(
    subset=[close_column]
)

data = data.reset_index(
    drop=True
)


# =========================================================
# DATA VALIDATION
# =========================================================

if len(data) < SEQUENCE_LENGTH:

    st.error(
        f"❌ At least {SEQUENCE_LENGTH} "
        "historical records are required."
    )

    st.stop()


# =========================================================
# HISTORICAL DATA
# =========================================================

st.subheader(
    "📋 Historical Stock Data"
)

st.dataframe(
    data.tail(10),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# PREPARE CLOSE PRICES
# =========================================================

prices = data[
    [close_column]
].values


# =========================================================
# SCALE DATA
# =========================================================

try:

    scaled_data = scaler.transform(
        prices
    )

except Exception as e:

    st.error(
        "❌ Error while scaling the data."
    )

    st.info(
        "The scaler must be the same scaler used "
        "during LSTM training."
    )

    st.exception(e)

    st.stop()


# =========================================================
# FORECAST FUNCTION
# =========================================================

def generate_forecast():

    sequence = (
        scaled_data[
            -SEQUENCE_LENGTH:
        ].copy()
    )

    predictions = []

    for _ in range(
        FORECAST_DAYS
    ):

        X = sequence.reshape(
            1,
            SEQUENCE_LENGTH,
            1
        )

        prediction = model.predict(
            X,
            verbose=0
        )

        predicted_value = float(
            prediction[0][0]
        )

        predictions.append(
            predicted_value
        )

        sequence = np.concatenate(
            [
                sequence[1:],
                np.array(
                    [[predicted_value]]
                )
            ],
            axis=0
        )

    predictions = np.array(
        predictions
    ).reshape(
        -1,
        1
    )

    predictions = scaler.inverse_transform(
        predictions
    )

    return predictions.flatten()


# =========================================================
# FORECAST BUTTON
# =========================================================

if st.button(
    "🚀 Generate 30-Day Forecast",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Generating LSTM forecast..."
    ):

        try:

            forecast = generate_forecast()

        except Exception as e:

            st.error(
                "❌ Forecast generation failed."
            )

            st.exception(e)

            st.stop()


    # =====================================================
    # LAST DATE
    # =====================================================

    if date_column is not None:

        last_date = data[
            date_column
        ].iloc[-1]

    else:

        last_date = pd.Timestamp.today()


    # =====================================================
    # FUTURE TRADING DAYS
    # =====================================================

    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=FORECAST_DAYS
    )


    # =====================================================
    # FORECAST DATAFRAME
    # =====================================================

    forecast_df = pd.DataFrame({

        "Date": future_dates,

        "Predicted Close Price": forecast

    })


    # =====================================================
    # FORECAST RESULTS
    # =====================================================

    st.subheader(
        "📊 Next 30 Trading-Day Forecast"
    )

    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # METRICS
    # =====================================================

    last_price = float(
        prices[-1][0]
    )

    day_1_price = float(
        forecast[0]
    )

    day_30_price = float(
        forecast[-1]
    )

    percentage_change = (
        (day_30_price - last_price)
        / last_price
    ) * 100


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Last Actual Price",
            f"₹{last_price:,.2f}"
        )


    with col2:

        st.metric(
            "Day 1 Forecast",
            f"₹{day_1_price:,.2f}"
        )


    with col3:

        st.metric(
            "Day 30 Forecast",
            f"₹{day_30_price:,.2f}",
            f"{percentage_change:.2f}%"
        )


    # =====================================================
    # CHART
    # =====================================================

    st.subheader(
        "📈 Historical vs LSTM Forecast"
    )

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    historical = data.tail(120)

    if date_column is not None:

        ax.plot(
            historical[date_column],
            historical[close_column],
            label="Historical Price"
        )

        ax.plot(
            future_dates,
            forecast,
            linestyle="--",
            label="LSTM Forecast"
        )

    else:

        historical_x = np.arange(
            len(historical)
        )

        forecast_x = np.arange(
            len(historical),
            len(historical) + FORECAST_DAYS
        )

        ax.plot(
            historical_x,
            historical[close_column],
            label="Historical Price"
        )

        ax.plot(
            forecast_x,
            forecast,
            linestyle="--",
            label="LSTM Forecast"
        )


    ax.set_title(
        "Reliance Stock Price Forecast using LSTM"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Stock Price (₹)"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(
        fig
    )


    # =====================================================
    # DOWNLOAD FORECAST
    # =====================================================

    csv_data = forecast_df.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv_data,
        file_name="reliance_30_day_forecast.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.success(
        "✅ 30-trading-day forecast generated successfully!"
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.subheader(
    "🤖 Model Information"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.info(
        "**Model**\n\nLSTM"
    )

with col2:

    st.info(
        "**Forecast Horizon**\n\n30 Trading Days"
    )

with col3:

    st.info(
        f"**Sequence Length**\n\n{SEQUENCE_LENGTH}"
    )


# =========================================================
# DEVELOPED BY
# ========================================================
# =========================================================
# DEVELOPED BY
# =========================================================

st.divider()

st.markdown(
    """
    ### 👨‍💻 Developed By

    **Dama Rajeswari**

    """
)

st.caption(
    "© 2026 Dama Rajeswari | Reliance Stock Price Forecasting"
)


# =========================================================
# DISCLAIMER
# =========================================================

st.warning(
    "⚠️ Disclaimer: This application is developed for "
    "educational and research purposes only. The forecasts "
    "are generated by a machine learning model and should "
    "not be considered financial or investment advice."
)
