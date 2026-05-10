import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ------------------------------------------------
# Page Config
# ------------------------------------------------
st.set_page_config(layout="wide")

st.title("Gold Price Forecast using ARIMA Model")
st.write("Forecast future gold prices using ARIMA.")

# ------------------------------------------------
# Load Model
# ------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("arima_gold_price_model.joblib")
    return model

# ------------------------------------------------
# Load CSV Data
# ------------------------------------------------
@st.cache_data
def load_data():

    train_df = pd.read_csv(
        "train_original.csv",
        parse_dates=True
    )

    test_df = pd.read_csv(
        "test_original.csv",
        parse_dates=True
    )

    return train_df, test_df

# ------------------------------------------------
# Load Resources
# ------------------------------------------------
model_train = load_model()

train_original, test_original = load_data()

# ------------------------------------------------
# Fix Date Columns
# ------------------------------------------------
# Convert first column to datetime safely
train_original[train_original.columns[0]] = pd.to_datetime(
    train_original[train_original.columns[0]],
    errors="coerce"
)

test_original[test_original.columns[0]] = pd.to_datetime(
    test_original[test_original.columns[0]],
    errors="coerce"
)

# Remove invalid rows
train_original.dropna(inplace=True)
test_original.dropna(inplace=True)

# Set first column as index
train_original.set_index(
    train_original.columns[0],
    inplace=True
)

test_original.set_index(
    test_original.columns[0],
    inplace=True
)

# ------------------------------------------------
# Select Gold Price Column
# ------------------------------------------------
train_values = train_original.iloc[:, 0]

test_values = test_original.iloc[:, 0]

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
st.sidebar.header("Forecast Settings")

forecast_days = st.sidebar.slider(
    "Forecast Days",
    min_value=1,
    max_value=365,
    value=60
)

# ------------------------------------------------
# Forecast Button
# ------------------------------------------------
if st.sidebar.button("Generate Forecast"):

    with st.spinner("Generating forecast..."):

        # Last date
        last_test_date = test_original.index.max()

        # Future start date
        future_start_date = last_test_date + pd.Timedelta(days=1)

        # Future business dates
        future_dates = pd.date_range(
            start=future_start_date,
            periods=forecast_days,
            freq="B"
        )

        # Forecast
        forecast_values = model_train.forecast(
            steps=forecast_days
        )

        # Create Series
        future_forecast = pd.Series(
            forecast_values,
            index=future_dates
        )

    # ------------------------------------------------
    # Forecast Table
    # ------------------------------------------------
    st.subheader("Forecast Results")

    forecast_df = pd.DataFrame({
        "Date": future_forecast.index,
        "Predicted Gold Price": future_forecast.values
    })

    st.dataframe(forecast_df)

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------
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
    
