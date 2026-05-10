
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")

st.title('Gold Price Forecast using ARIMA Model')
st.write('This application forecasts gold prices for a specified number of future days.')

# Load the trained model
@st.cache_resource
def load_model(filename):
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{filename}' not found. Make sure it's in the same directory.")
        st.stop()

# Load the original training and test data
@st.cache_data
def load_data():
    try:
        train_df = pd.read_csv('train_original.csv', index_col=0, parse_dates=True)
        test_df = pd.read_csv('test_original.csv', index_col=0, parse_dates=True)
        return train_df, test_df
    except FileNotFoundError:
        st.error("Error: Data files 'train_original.csv' or 'test_original.csv' not found. Make sure they are in the same directory.")
        st.stop()

model_train = load_model('arima_gold_price_model.joblib')
train_original, test_original = load_data()

if model_train is None or train_original is None or test_original is None:
    st.stop()

st.sidebar.header('Forecast Settings')
forecast_days = st.sidebar.slider('Number of days to forecast', 1, 365, 60)

if st.sidebar.button('Generate Forecast'):
    with st.spinner(f'Generating forecast for {forecast_days} days...'):
        # Determine the start date for the future forecast
        last_test_date = test_original.index.max()
last_test_date = pd.to_datetime(last_test_date)    
future_forecast_start_date = last_test_date + pd.Timedelta(days=1)

        # Create a date range for these future business days
        future_dates = pd.date_range(start=future_forecast_start_date, periods=forecast_days, freq='B')
        future_forecast_end_date = future_dates.max()

        # Generate forecast
        future_forecast_original = model_train.predict(start=future_forecast_start_date, end=future_forecast_end_date)
        
        # Ensure the index of the forecast is correct (pmdarima's predict with dates usually handles this)
        if not isinstance(future_forecast_original.index, pd.DatetimeIndex):
            future_forecast_original = pd.Series(future_forecast_original, index=future_dates)

    st.subheader(f'Gold Price Forecast for the Next {forecast_days} Business Days')
    st.write(future_forecast_original)

    st.subheader('Forecast Visualization')
    fig = plt.figure(figsize=(15, 7))
    plt.plot(train_original.index, train_original, label='Training Data (Last 5 Years)', color='blue')
    plt.plot(test_original.index, test_original, label='Actual Test Data (Last 60 Days)', color='orange')
    
    # Plot previous forecast if it was generated (this is `forecast_original` from the notebook's test set)
    # For simplicity in a fresh Streamlit app, we might re-forecast the test set or just plot actuals.
    # Given `test_original` is loaded, we can plot previous forecasts directly if needed, but for future focus, omit.

    plt.plot(future_forecast_original.index, future_forecast_original, label=f'Future {forecast_days}-Day Forecast', color='red', linestyle='-.')

    plt.title('Gold Price Forecast: Actual vs. Predicted & Future Forecast')
    plt.xlabel('Date')
    plt.ylabel('Gold Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

    st.success('Forecast generated successfully!')
else:
    st.info('Adjust the number of days and click "Generate Forecast" to see predictions.')
