# === STEP 1: IMPORT LIBRARIES === #
import pandas as pd
import numpy as np
from prophet import Prophet
import holidays
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from google.colab import files
import io

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# === STEP 2: LOAD & CLEAN DATA === #
uploaded = files.upload()
filename = list(uploaded.keys())[0]
data_raw = pd.read_csv(io.BytesIO(uploaded[filename]))

data_raw['Date'] = pd.to_datetime(data_raw['Date'], dayfirst=True, errors='coerce')
data_raw = data_raw[['Date', 'Open', 'High', 'Low', 'Close']]
data_raw.dropna(inplace=True)

# Remove duplicates
data_raw = data_raw.drop_duplicates(subset=['Date'])

# Ensure numeric
data_raw['Close'] = pd.to_numeric(data_raw['Close'], errors='coerce')
data_raw.dropna(inplace=True)

# Sort
data_raw = data_raw.sort_values('Date')

# === STEP 3: OUTLIER CAPPING === #
q1, q99 = data_raw['Close'].quantile([0.01, 0.99])
data_raw['Close'] = data_raw['Close'].clip(lower=q1, upper=q99)


# === STEP 4: HOLIDAY CALENDAR === #
years = pd.DatetimeIndex(data_raw["Date"]).year.unique()
ind_holidays = holidays.India(years=years)
holiday_df = pd.DataFrame({
    "ds": pd.to_datetime(list(ind_holidays.keys())),
    "holiday": "india_national"
})

# === STEP 5: FEATURE ENGINEERING === #
def create_features(df, log_transform=None):
    df = df.copy()
    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

    if log_transform == "log10":
        df['y'] = np.log10(df['y'])

    df['diff_1'] = df['y'].diff()
    df['y_lag1'] = df['y'].shift(1)
    df['Daily_Return'] = df['y'].pct_change()
    df['MA_50'] = df['y'].rolling(50).mean()
    df['MA_200'] = df['y'].rolling(200).mean()
    df['Volatility'] = df['Daily_Return'].rolling(30).std()

    df.dropna(inplace=True)
    return df


df_raw = create_features(data_raw)
df_log10 = create_features(data_raw, log_transform="log10")


# === STEP 6: PROPHET === #
def train_prophet(df, holidays, log=None):
    model = Prophet(holidays=holidays)

    for reg in ['y_lag1','MA_50','MA_200','Volatility']:
        if reg in df.columns:
            model.add_regressor(reg)

    model.fit(df)

    future = model.make_future_dataframe(periods=30)
    merged = pd.merge(future, df[['ds','y_lag1','MA_50','MA_200','Volatility']], on='ds', how='left')
    merged.fillna(method='ffill', inplace=True)

    forecast = model.predict(merged)

    if log == "log10":
        forecast['yhat'] = np.power(10, forecast['yhat'])

    eval_df = forecast.merge(df[['ds','y']], on='ds')

    mape = mean_absolute_percentage_error(eval_df['y'], eval_df['yhat']) * 100
    rmse = np.sqrt(mean_squared_error(eval_df['y'], eval_df['yhat']))

    return mape, rmse


mape_prophet_raw, rmse_prophet_raw = train_prophet(df_raw, holiday_df)
mape_prophet_log10, rmse_prophet_log10 = train_prophet(df_log10, holiday_df, log="log10")


# === STEP 7: XGBOOST === #
def train_xgb(df):
    features = ['y_lag1','Daily_Return','MA_50','MA_200','Volatility']
    X = df[features].fillna(0)
    y = df['y']

    split = int(len(df)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, y_pred)*100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return mape, rmse


mape_xgb, rmse_xgb = train_xgb(df_raw)
mape_xgb_log10, rmse_xgb_log10 = train_xgb(df_log10)

# === STEP 8: ARIMA === #
def train_arima(df):
    series = df['y']
    split = int(len(series)*0.8)

    train, test = series[:split], series[split:]

    model = ARIMA(train, order=(5,1,0)).fit()
    forecast = model.forecast(len(test))

    mape = mean_absolute_percentage_error(test, forecast)*100
    rmse = np.sqrt(mean_squared_error(test, forecast))

    return mape, rmse


mape_arima, rmse_arima = train_arima(df_raw)
mape_arima_log10, rmse_arima_log10 = train_arima(df_log10)

# === STEP 9: LSTM === #
def train_lstm(df, look_back=30):
    from sklearn.preprocessing import MinMaxScaler

    series = df['y'].values.reshape(-1,1)
    split = int(len(series)*0.8)

    train, test = series[:split], series[split:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    def create_seq(data):
        X, y = [], []
        for i in range(len(data)-look_back):
            X.append(data[i:i+look_back])
            y.append(data[i+look_back])
        return np.array(X), np.array(y)

    X_train, y_train = create_seq(train_scaled)
    X_test, y_test = create_seq(test_scaled)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(look_back,1)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0)

    y_pred = model.predict(X_test)

    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)

    mape = mean_absolute_percentage_error(y_test, y_pred)*100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return mape, rmse


mape_lstm, rmse_lstm = train_lstm(df_raw)
mape_lstm_log10, rmse_lstm_log10 = train_lstm(df_log10)


# === STEP 10: FINAL TABLE === #
comparison = pd.DataFrame({
    "Model": ["Prophet", "ARIMA", "XGBoost", "LSTM"],

    "MAPE Raw": [mape_prophet_raw, mape_arima, mape_xgb, mape_lstm],
    "MAPE Log10": [mape_prophet_log10, mape_arima_log10, mape_xgb_log10, mape_lstm_log10],

    "RMSE Raw": [rmse_prophet_raw, rmse_arima, rmse_xgb, rmse_lstm],
    "RMSE Log10": [rmse_prophet_log10, rmse_arima_log10, rmse_xgb_log10, rmse_lstm_log10]
})

print("\n📊 FINAL MODEL COMPARISON:\n")
print(comparison)

# === STEP 11: XGBOOST 30-DAY FORECAST === #

def forecast_xgb_30_days(df, model, steps=15):
    df_future = df.copy()

    forecasts = []

    for _ in range(steps):
        last_row = df_future.iloc[-1:].copy()

        X = last_row[['y_lag1','Daily_Return','MA_50','MA_200','Volatility']].fillna(0)
        y_pred = model.predict(X)[0]

        # Create next row
        next_row = last_row.copy()
        next_row['ds'] = next_row['ds'] + pd.Timedelta(days=1)
        next_row['y'] = y_pred

        # Recompute features
        df_temp = pd.concat([df_future, next_row], ignore_index=True)

        df_temp['y_lag1'] = df_temp['y'].shift(1)
        df_temp['Daily_Return'] = df_temp['y'].pct_change()
        df_temp['MA_50'] = df_temp['y'].rolling(50).mean()
        df_temp['MA_200'] = df_temp['y'].rolling(200).mean()
        df_temp['Volatility'] = df_temp['Daily_Return'].rolling(30).std()

        df_future = df_temp.copy()

        forecasts.append({
            "Date": next_row['ds'].values[0],
            "Predicted_Close": y_pred
        })

    forecast_df = pd.DataFrame(forecasts)
    return forecast_df


# === TRAIN FINAL XGBOOST MODEL ON FULL DATA === #
features = ['y_lag1','Daily_Return','MA_50','MA_200','Volatility']
X_full = df_raw[features].fillna(0)
y_full = df_raw['y']

final_xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
final_xgb_model.fit(X_full, y_full)


# === GENERATE FORECAST === #
forecast_30_days = forecast_xgb_30_days(df_raw, final_xgb_model, steps=30)

print("\n📈 XGBoost 30-Day Forecast:\n")
print(forecast_30_days)

# === STEP 12: WALK-FORWARD FORECAST (30 DAYS) === #

def walk_forward_xgb(df, steps=30):
    df_wf = df.copy()
    predictions = []

    for i in range(steps):
        # Train model on current available data
        features = ['y_lag1','Daily_Return','MA_50','MA_200','Volatility']
        X = df_wf[features].fillna(0)
        y = df_wf['y']

        model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
        model.fit(X, y)

        # Predict next step
        last_row = df_wf.iloc[-1:].copy()
        X_last = last_row[features].fillna(0)
        y_pred = model.predict(X_last)[0]

        # Create next row
        next_row = last_row.copy()
        next_row['ds'] = next_row['ds'] + pd.Timedelta(days=1)
        next_row['y'] = y_pred

        # Append temporarily
        df_temp = pd.concat([df_wf, next_row], ignore_index=True)

        # Recompute features
        df_temp['y_lag1'] = df_temp['y'].shift(1)
        df_temp['Daily_Return'] = df_temp['y'].pct_change()
        df_temp['MA_50'] = df_temp['y'].rolling(50).mean()
        df_temp['MA_200'] = df_temp['y'].rolling(200).mean()
        df_temp['Volatility'] = df_temp['Daily_Return'].rolling(30).std()

        df_wf = df_temp.copy()

        predictions.append({
            "Step": i+1,
            "Date": next_row['ds'].values[0],
            "Predicted_Close": y_pred
        })

    return pd.DataFrame(predictions)


# === RUN WALK-FORWARD === #
walk_forward_30 = walk_forward_xgb(df_raw, steps=30)

print("\n📊 Walk-Forward Forecast (30 Days):\n")
print(walk_forward_30)


