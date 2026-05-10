import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------
# Streamlit Page
# ---------------------------------------
st.set_page_config(layout="wide")

st.title("Gold Price Forecast using ARIMA Model")
st.write("Forecast future gold prices using ARIMA.")

# ---------------------------------------
# Load Model
# ---------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("arima_gold_price_model.joblib")

# ---------------------------------------
# Load Data
# ---------------------------------------
@st.cache_data
def load_data():

    # Read CSV files
    train_df = pd.read_csv("train_original.csv")
    test_df = pd.read_csv("test_original.csv")

    return train_df, test_df

# ---------------------------------------
# Load Resources
# ---------------------------------------
model_train = load_model()

train_original, test_original = load_data()

# ---------------------------------------
# Fix Date Column
# ---------------------------------------
# Remove unnamed column if exists
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

# Convert Date column
train_original["Date"] = pd.to_datetime(
    train_original["Date"],
    errors="coerce"
)

test_original["Date"] = pd.to_datetime(
    test_original["Date"],
    errors="coerce"
)

# Remove bad rows
train_original.dropna(inplace=True)
test_original.dropna(inplace=True)

# Set date index
train_original.set_index("Date", inplace=True)
test_original.set_index("Date", inplace=True)

# ---------------------------------------
# Select Gold Price Column
# ---------------------------------------
train_values = train_original.iloc[:, 0]
test_values = test_original.iloc[:, 0]

# ---------------------------------------
# Sidebar
# ---------------------------------------
st.sidebar.header("Forecast Settings")

forecast_days = st.sidebar.slider(
    "Forecast Days",
    1,
    365,
    60
)

# ---------------------------------------
# Forecast
# ---------------------------------------
if st.sidebar.button("Generate Forecast"):

    with st.spinner("Generating forecast..."):

        # Forecast future values
        forecast_values = model_train.forecast(
            steps=forecast_days
        )

        # Last available date
        last_date = test_original.index.max()

        # Future dates
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="B"
        )

        # Forecast series
        future_forecast = pd.Series(
            forecast_values,
            index=future_dates
        )

    # ---------------------------------------
    # Forecast Table
    # ---------------------------------------
    st.subheader("Forecast Results")

    forecast_df = pd.DataFrame({
        "Date": future_forecast.index,
        "Predicted Gold Price": future_forecast.values
    })

    st.dataframe(forecast_df)

    # ---------------------------------------
    # Plot
    # ---------------------------------------
    st.subheader("Forecast Visualization")

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(
        train_values.index,
        train_values.values,
        label="Training Data"
    )

    ax.plot(
        test_values.index,
        test_values.values,
        label="Test Data"
    )

    ax.plot(
        future_forecast.index,
        future_forecast.values,
        linestyle="--",
        label="Future Forecast"
    )

    ax.set_title("Gold Price Forecast")

    ax.set_xlabel("Date")

    ax.set_ylabel("Gold Price")

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)

    st.success("Forecast generated successfully!")

else:
    st.info("Select forecast days and click Generate Forecast.")
