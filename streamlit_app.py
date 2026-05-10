import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title('Gold Price Forecast using ARIMA Model')
st.write('This application forecasts gold prices for a specified number of future days.')

# Load trained model
@st.cache_resource
def load_model(filename):
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{filename}' not found.")
        st.stop()

# Load data
@st.cache_data
def load_data():
    try:
        train_df = pd.read_csv(
            'train_original.csv',
            index_col=0,
            parse_dates=True
        )

        test_df = pd.read_csv(
            'test_original.csv',
            index_col=0,
            parse_dates=True
        )

        return train_df, test_df

    except FileNotFoundError:
        st.error("Data files not found.")
        st.stop()

# Load model and datasets
model_train = load_model('arima_gold_price_model.joblib')
train_original, test_original = load_data()

# Sidebar settings
st.sidebar.header('Forecast Settings')

forecast_days = st.sidebar.slider(
    'Number of days to forecast',
    min_value=1,
    max_value=365,
    value=60
)

# Generate forecast
if st.sidebar.button('Generate Forecast'):

    with st.spinner(f'Generating forecast for {forecast_days} days...'):

        # Get last date from test data
        last_test_date = pd.to_datetime(test_original.index.max())

        # Next forecast start date
        future_forecast_start_date = last_test_date + pd.Timedelta(days=1)

        # Create future business dates
        future_dates = pd.date_range(
            start=future_forecast_start_date,
            periods=forecast_days,
            freq='B'
        )

        # Forecast values
        forecast_values = model_train.forecast(steps=forecast_days)

        # Convert forecast to pandas Series
        future_forecast_original = pd.Series(
            forecast_values,
            index=future_dates
        )

    # Show forecast table
    st.subheader(f'Gold Price Forecast for Next {forecast_days} Business Days')

    st.write(future_forecast_original)

    # Plot graph
    st.subheader('Forecast Visualization')

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(
        train_original.index,
        train_original.values,
        label='Training Data',
    )

    ax.plot(
        test_original.index,
        test_original.values,
        label='Test Data',
    )

    ax.plot(
        future_forecast_original.index,
        future_forecast_original.values,
        label='Future Forecast',
        linestyle='--'
    )

    ax.set_title('Gold Price Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Gold Price')

    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.success('Forecast generated successfully!')

else:
    st.info('Adjust settings and click "Generate Forecast".')
