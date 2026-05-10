import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Streamlit Page Settings
# -------------------------------
st.set_page_config(layout="wide")

st.title("Gold Price Forecast using ARIMA Model")
st.write("This application forecasts gold prices for future business days.")

# -------------------------------
# Load Trained Model
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("arima_gold_price_model.joblib")
        return model

    except FileNotFoundError:
        st.error("Model file not found.")
        st.stop()

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():

    try:
        train_df = pd.read_csv("train_original.csv")
        test_df = pd.read_csv("test_original.csv")

        return train_df, test_df

    except FileNotFoundError:
        st.error("CSV files not found.")
        st.stop()

# -------------------------------
# Load Resources
# -------------------------------
model_train = load_model()

train_original, test_original = load_data()

# -------------------------------
# Convert Date Columns
# -------------------------------
# Automatically detect first column as date
train_original.iloc[:, 0] = pd.to_datetime(
    train_original.iloc[:, 0],
    errors="coerce"
)

test_original.iloc[:, 0] = pd.to_datetime(
    test_original.iloc[:, 0],
    errors="coerce"
)

# Remove invalid rows
train_original = train_original.dropna()
test_original = test_original.dropna()

# Set date as index
train_original.set_index(train_original.columns[0], inplace=True)
test_original.set_index(test_original.columns[0], inplace=True)

# -------------------------------
# Select Gold Price Column
# -------------------------------
train_values = train_original.iloc[:, 0]
test_values = test_original.iloc[:, 0]

# -------------------------------
# Sidebar Settings
# -------------------------------
st.sidebar.header("Forecast Settings")

forecast_days = st.sidebar.slider(
    "Number of days to forecast",
    min_value=1,
    max_value=365,
    value=60
)

# -------------------------------
# Forecast Button
# -------------------------------
if st.sidebar.button("Generate Forecast"):

    with st.spinner("Generating forecast..."):

        # Get last available date
        last_test_date = test_original.index.max()

        # Future start date
        future_forecast_start_date = last_test_date + pd.Timedelta(days=1)

        # Create future business dates
        future_dates = pd.date_range(
            start=future_forecast_start_date,
            periods=forecast_days,
            freq="B"
        )

        # Forecast future values
        forecast_values = model_train.forecast(steps=forecast_days)

        # Convert forecast to series
        future_forecast = pd.Series(
            forecast_values,
            index=future_dates
        )

    # -------------------------------
    # Show Forecast Table
    # -------------------------------
    st.subheader(f"Forecast for Next {forecast_days} Business Days")

    forecast_df = pd.DataFrame({
        "Date": future_forecast.index,
        "Predicted Gold Price": future_forecast.values
    })

    st.dataframe(forecast_df)

    # -------------------------------
    # Plot Forecast
    # -------------------------------
    st.subheader("Forecast Visualization")

    fig, ax = plt.subplots(figsize=(15, 7))

    # Training data
    ax.plot(
        train_values.index,
        train_values.values,
        label="Training Data"
    )

    # Test data
    ax.plot(
        test_values.index,
        test_values.values,
        label="Test Data"
    )

    # Forecast data
    ax.plot(
        future_forecast.index,
        future_forecast.values,
        label="Future Forecast",
        linestyle="--"
    )

    ax.set_title("Gold Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Gold Price")

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)

    st.success("Forecast generated successfully!")

else:
    st.info("Adjust settings and click Generate Forecast.")
