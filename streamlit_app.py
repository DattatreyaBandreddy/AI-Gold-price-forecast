import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------
st.set_page_config(
    page_title="Gold Price Forecast",
    layout="wide"
)

st.title("Gold Price Forecast using ARIMA Model")
st.write("Forecast future gold prices using Time Series Forecasting.")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("arima_gold_price_model.joblib")
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():

    try:
        train_df = pd.read_csv("train_original.csv")
        test_df = pd.read_csv("test_original.csv")

        return train_df, test_df

    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        st.stop()

# ---------------------------------------------------
# LOAD FILES
# ---------------------------------------------------
model_train = load_model()

train_original, test_original = load_data()

# ---------------------------------------------------
# FIX DATE COLUMNS
# ---------------------------------------------------
# Rename unnamed columns
if "Unnamed: 0" in train_original.columns:
    train_original.rename(
        columns={"Unnamed: 0": "Date"},
        inplace=True
    )

if "Unnamed: 0" in test_original.columns:
    test_original.rename(
        columns={"Unnamed: 0": "Date"},
        inplace=True
    )

# Convert to datetime
train_original["Date"] = pd.to_datetime(
    train_original["Date"],
    errors="coerce"
)

test_original["Date"] = pd.to_datetime(
    test_original["Date"],
    errors="coerce"
)

# Remove invalid rows
train_original.dropna(inplace=True)
test_original.dropna(inplace=True)

# Set index
train_original.set_index("Date", inplace=True)
test_original.set_index("Date", inplace=True)

# ---------------------------------------------------
# SELECT TARGET COLUMN
# ---------------------------------------------------
train_values = train_original.iloc[:, 0]
test_values = test_original.iloc[:, 0]

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("Forecast Settings")

forecast_days = st.sidebar.slider(
    "Select Number of Forecast Days",
    min_value=1,
    max_value=365,
    value=60
)

# ---------------------------------------------------
# GENERATE FORECAST
# ---------------------------------------------------
if st.sidebar.button("Generate Forecast"):

    with st.spinner("Generating forecast..."):

        try:

            # Forecast future values
            forecast_values = model_train.forecast(
                steps=forecast_days
            )

            # Convert forecast to numpy array
            forecast_values = np.array(
                forecast_values
            ).flatten()

            # Last available date
            last_date = test_original.index.max()

            # Future business dates
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_days,
                freq="B"
            )

            # Create forecast series
            future_forecast = pd.Series(
                forecast_values,
                index=future_dates
            )

            # ---------------------------------------------------
            # SHOW FORECAST TABLE
            # ---------------------------------------------------
            st.subheader("Forecast Results")

            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted Gold Price": forecast_values
            })

            st.dataframe(
                forecast_df,
                use_container_width=True
            )

            # ---------------------------------------------------
            # PLOT GRAPH
            # ---------------------------------------------------
            st.subheader("Forecast Visualization")

            fig, ax = plt.subplots(
                figsize=(16, 8)
            )

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

            # Forecast
            ax.plot(
                future_forecast.index,
                future_forecast.values,
                linestyle="--",
                linewidth=2,
                label="Future Forecast"
            )

            ax.set_title(
                "Gold Price Forecast using ARIMA"
            )

            ax.set_xlabel("Date")

            ax.set_ylabel("Gold Price")

            ax.legend()

            ax.grid(True)

            st.pyplot(fig)

            # ---------------------------------------------------
            # SUCCESS MESSAGE
            # ---------------------------------------------------
            st.success(
                "Forecast generated successfully!"
            )

        except Exception as e:
            st.error(f"Forecast Error: {e}")

else:
    st.info(
        "Select forecast days and click Generate Forecast."
    )
