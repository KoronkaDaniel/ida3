import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
import warnings
import os

warnings.filterwarnings("ignore")

from ida3_conf import *


def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def add_lag_features(df, lag_cols, lags):
    for col in lag_cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df, rolling_cols, window_sizes):
    for col in rolling_cols:
        for window in window_sizes:
            df[f"{col}_roll_mean_{window}"] = df[col].rolling(window=window, min_periods=1).mean()
            df[f"{col}_roll_std_{window}"] = df[col].rolling(window=window, min_periods=1).std()
    print(f"Added rolling features for: {rolling_cols}")
    return df


def add_datetime_features(df):
    df['hour'] = df['ts_start'].dt.hour
    df['dayofweek'] = df['ts_start'].dt.dayofweek
    df['is_weekend'] = (df['ts_start'].dt.dayofweek >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df


def add_delta_features(df):
    if 'ida1' in df.columns and 'ida2' in df.columns:
        df['ida2_minus_ida1'] = df['ida2'] - df['ida1']
        df['ida1_ida2_ratio'] = df['ida2'] / (df['ida1'] + 1e-6)
    return df


def load_and_prepare(path):
    df = pd.read_csv(path, parse_dates=['ts_start', 'ts_end'])
    df.sort_values('ts_start', inplace=True)
    df.set_index('ts_start', inplace=True, drop=False)

    original_training_mask = df[TARGET_COLUMN].notna()

    df = add_delta_features(df)
    if USE_ROLLING_FEATURES:
        df = add_rolling_features(df, ROLLING_FEATURES_COLS, ROLLING_WINDOW_SIZES)
    df = add_lag_features(df, LAG_FEATURES, LAG_STEPS)
    if USE_DATE_FEATURES:
        df = add_datetime_features(df)

    training_data = df[original_training_mask].copy()

    print(f"Training data before cleaning NAs: {len(training_data)} rows")
    training_data.fillna(0, inplace=True)
    print(f"Training data after cleaning NAs: {len(training_data)} rows")
    
    return df, training_data


def get_backtest_period(df):
    """Find a suitable historical 12:00-00:00 period for backtesting"""
    latest_date_with_data = df.index.max().date()
    for i in range(2, 10):
        test_date = latest_date_with_data - timedelta(days=i)
        test_start = pd.to_datetime(f"{test_date} 12:00:00")
        test_end = test_start + timedelta(hours=11, minutes=45)
        
        test_mask = (df.index >= test_start) & (df.index <= test_end)
        
        if test_mask.sum() == N_PREDICT:
            print(f"Found suitable backtest period: {test_date}")
            return test_date, test_mask
    
    raise ValueError("Could not find a suitable, contiguous 12-hour block for backtesting.")


def train_robust_xgb(df, features, target):
    X = df[features]
    y = df[target]

    if len(df) < 50:
        raise ValueError(f"Not enough training data ({len(df)} rows) to train.")

    val_size = int(len(df) * 0.25)
    
    X_train, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
    y_train, y_val = y.iloc[:-val_size], y.iloc[-val_size:]
    print(f"Train/Validation split: {len(X_train)} training, {len(X_val)} validation points.")

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective='reg:squarederror',
        tree_method='hist',
        eval_metric='mae',
        early_stopping_rounds=50
    )
    print("Training model...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def prepare_forecast_data(df, forecast_start, n_periods):
    forecast_end = forecast_start + timedelta(minutes=15 * n_periods)
    forecast_mask = (df['ts_start'] >= forecast_start) & (df['ts_start'] < forecast_end)
    forecast_data = df[forecast_mask].copy()

    if len(forecast_data) == 0:
        return None
        
    if forecast_data.isnull().values.any():
        print("⚠️ WARNING: Found NaN values in forecast data slice. Filling with 0.")
        forecast_data.fillna(0, inplace=True)
    return forecast_data


if __name__ == "__main__":
    PATH = "/home/bzsolt/Arthemis ML/q_hourly_db_cleaned.csv"
    
    if not os.path.exists(PATH):
        raise FileNotFoundError(f"CSV file not found at: {PATH}")
    
    df_all, df_training = load_and_prepare(PATH)

    feature_list = NUMERIC_FEATURES.copy()
    feature_list.extend(add_delta_features(pd.DataFrame(columns=df_all.columns)).columns.tolist())
    if USE_ROLLING_FEATURES:
        feature_list.extend([f for f in df_all.columns if "_roll_" in f])
    feature_list.extend([f for f in df_all.columns if "_lag" in f])
    if USE_DATE_FEATURES:
        feature_list.extend(['hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos'])

    available_features = sorted(list(set(feature_list) & set(df_training.columns)))
    
    if TARGET_COLUMN in available_features: available_features.remove(TARGET_COLUMN)
    if 'ts_start' in available_features: available_features.remove('ts_start')
    if 'ts_end' in available_features: available_features.remove('ts_end')
        
    print(f"\nUsing {len(available_features)} features for training.")

    # --- 1. BACKTESTING ---
    print("\n" + "="*60 + "\nBACKTESTING ON HISTORICAL DATA\n" + "="*60)
    backtest_results = {}
    try:
        backtest_date, backtest_mask = get_backtest_period(df_training)
        
        train_for_backtest = df_training[~backtest_mask]
        backtest_data = df_training[backtest_mask]

        model_backtest = train_robust_xgb(train_for_backtest, available_features, TARGET_COLUMN)
        
        backtest_preds = model_backtest.predict(backtest_data[available_features])
        backtest_actuals = backtest_data[TARGET_COLUMN].values
        
        backtest_mae = mean_absolute_error(backtest_actuals, backtest_preds)
        backtest_smape = smape(backtest_actuals, backtest_preds)
        print(f"\n🎯 Backtest Results for {backtest_date}: MAE={backtest_mae:.3f}, sMAPE={backtest_smape:.2f}%")

        backtest_results = {
            "preds": backtest_preds, "actuals": backtest_actuals,
            "data": backtest_data, "date": backtest_date,
            "mae": backtest_mae, "smape": backtest_smape
        }

    except Exception as e:
        print(f"❌ Backtesting failed: {e}")

    # --- 2. FORECAST ---
    print("\n" + "="*60 + f"\nFORECASTING FOR {FORECAST_DATE}\n" + "="*60)
    forecast_results = {}
    if df_training.empty:
        print("❌ ERROR: No training data available. Cannot train model.")
    else:
        model_forecast = train_robust_xgb(df_training, available_features, TARGET_COLUMN)
        
        forecast_start = pd.to_datetime(f"{FORECAST_DATE} 12:00:00")
        forecast_data = prepare_forecast_data(df_all, forecast_start, N_PREDICT)

        if forecast_data is not None and not forecast_data.empty:
            print(f"\n🔮 Generating {len(forecast_data)} predictions...")
            forecast_preds = model_forecast.predict(forecast_data[available_features])
            
            forecast_results = {"preds": forecast_preds, "data": forecast_data}
            print("\nForecast Results:")
            print(pd.DataFrame({'timestamp': forecast_data['ts_start'].values, 'predicted_ida3': forecast_preds}))

    # --- 3. VISUALIZATION ---
    print("\n" + "="*60 + "\nGENERATING PLOTS\n" + "="*60)
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=False)
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: Backtesting Results
    if backtest_results:
        br = backtest_results
        axes[0].plot(br['data'].index, br['actuals'], label="Actual IDA3", color='blue', linewidth=2)
        axes[0].plot(br['data'].index, br['preds'], label="Predicted IDA3", color='red', linestyle='--', linewidth=2)
        axes[0].plot(br['data'].index, br['data']['ida1'], label="IDA1", color='green', alpha=0.6)
        axes[0].plot(br['data'].index, br['data']['ida2'], label="IDA2", color='orange', alpha=0.6)
        axes[0].set_title(f"Backtesting Results - {br['date']}\nMAE: {br['mae']:.3f}, sMAPE: {br['smape']:.2f}%", fontsize=14)
        axes[0].set_ylabel("Price")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, 'Backtesting not available', horizontalalignment='center', transform=axes[0].transAxes)
        axes[0].set_title("Backtesting Results")

    # Plot 2: Forecast Results
    if forecast_results:
        fr = forecast_results
        axes[1].plot(fr['data'].index, fr['preds'], label="Predicted IDA3", color='red', linestyle='--', marker='o', markersize=4)
        axes[1].plot(fr['data'].index, fr['data']['ida1'], label="IDA1 (Input)", color='green', alpha=0.7)
        axes[1].plot(fr['data'].index, fr['data']['ida2'], label="IDA2 (Input)", color='blue', alpha=0.7)
        
        last_day_of_training = df_training.tail(48)
        axes[1].plot(last_day_of_training.index, last_day_of_training[TARGET_COLUMN], label="Previous Day IDA3", color='gray', alpha=0.8)
        
        axes[1].set_title(f"Forecast Results - {FORECAST_DATE}", fontsize=14)
        axes[1].set_ylabel("Price")
        axes[1].legend()

        # --- MODIFICATION: Set x-axis limits to the forecast period ---
        axes[1].set_xlim(fr['data'].index.min(), fr['data'].index.max())
    else:
        axes[1].text(0.5, 0.5, 'Forecast not available', horizontalalignment='center', transform=axes[1].transAxes)
        axes[1].set_title(f"Forecast Results - {FORECAST_DATE}")

    fig.tight_layout(pad=3.0)
    plt.show()