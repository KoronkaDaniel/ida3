import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
import warnings
import os
import json
import optuna
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

from ida3_conf import *

# --- Functions (No Changes) ---
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
def add_lag_features(df, lag_cols, lags):
    for col in lag_cols:
        for lag in lags: df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df
def add_rolling_features(df, rolling_cols, window_sizes):
    for col in rolling_cols:
        for window in window_sizes:
            df[f"{col}_roll_mean_{window}"] = df[col].rolling(window=window, min_periods=1).mean()
            df[f"{col}_roll_std_{window}"] = df[col].rolling(window=window, min_periods=1).std()
    print(f"Added rolling features for: {rolling_cols}")
    return df
def add_datetime_features(df):
    df['hour'], df['dayofweek'] = df['ts_start'].dt.hour, df['ts_start'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['hour_sin'], df['hour_cos'] = np.sin(2*np.pi*df['hour']/24), np.cos(2*np.pi*df['hour']/24)
    df['time_in_session'] = ((df.index.hour - 12) * 4 + (df.index.minute/15)).where(df.index.hour >= 12, 0)
    return df
def add_delta_features(df):
    if 'ida1' in df.columns and 'ida2' in df.columns: df['ida2_minus_ida1'] = df['ida2'] - df['ida1']
    if 'dam' in df.columns and 'ida1' in df.columns: df['ida1_minus_dam'] = df['ida1'] - df['dam']
    return df
def load_and_prepare(path):
    df = pd.read_csv(path, parse_dates=['ts_start', 'ts_end'])
    df.sort_values('ts_start', inplace=True); df.set_index('ts_start', inplace=True, drop=False)
    df = add_delta_features(df)
    if USE_ROLLING_FEATURES: df = add_rolling_features(df, ROLLING_FEATURES_COLS, ROLLING_WINDOW_SIZES)
    if USE_LAG_FEATURES: df = add_lag_features(df, LAG_FEATURES, LAG_STEPS)
    if USE_DATE_FEATURES: df = add_datetime_features(df)
    if MODELING_STRATEGY == 'delta':
        df['target_delta'] = df[TARGET_COLUMN] - df[DELTA_BASE_COLUMN]
        training_mask = df['target_delta'].notna()
    else: training_mask = df[TARGET_COLUMN].notna()
    training_data = df[training_mask].copy()
    print(f"Training data before cleaning NAs: {len(training_data)} rows")
    training_data.fillna(0, inplace=True); print(f"Training data after cleaning NAs: {len(training_data)} rows")
    return df, training_data
def run_hpo(X_train, y_train, X_val, y_val, model_objective, quantile_alpha=0.5):
    def objective(trial):
        params = {
            'objective': model_objective, 'tree_method': 'hist', 'n_estimators': 2000, 'random_state': 42,
            'early_stopping_rounds': 50,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
        if model_objective == 'reg:quantileerror': params['quantile_alpha'] = quantile_alpha
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return mean_absolute_error(y_val, model.predict(X_val))
    study = optuna.create_study(direction='minimize')
    n_trials = 50
    desc = f"HPO for P{int(quantile_alpha*100)}" if model_objective == 'reg:quantileerror' else "HPO for Residuals"
    with tqdm(total=n_trials, desc=desc) as pbar:
        def tqdm_callback(study, trial): pbar.update(1)
        try: study.optimize(objective, n_trials=n_trials, callbacks=[tqdm_callback], gc_after_trial=True)
        except Exception as e: print(f"An error occurred during HPO: {e}")
    if study.best_trial: return study.best_params
    return {}
def train_quantile_model(df, features, target, quantile, use_hpo=False, fit_residuals=False):
    X = df[features]; y = df[target]
    if len(df) < 50: raise ValueError(f"Not enough data ({len(df)} rows) to train.")
    val_size = int(len(df) * 0.25)
    X_train, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
    y_train, y_val = y.iloc[:-val_size], y.iloc[-val_size:]
    quantile_key = f'p{int(quantile*100)}'
    print(f"\nTraining {quantile_key} model... Train/Val split: {len(X_train)}/{len(X_val)}")
    params = {}
    hpo_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), HPO_RESULTS_FILE)
    if use_hpo:
        all_hpo_params = {}
        if os.path.exists(hpo_file_path):
            with open(hpo_file_path, 'r') as f: all_hpo_params = json.load(f)
        if quantile_key in all_hpo_params:
            print(f"Loading best params for {quantile_key} from {HPO_RESULTS_FILE}")
            params = all_hpo_params[quantile_key]
        else:
            print(f"No params found for {quantile_key}. Running new HPO study...")
            params = run_hpo(X_train, y_train, X_val, y_val, 'reg:quantileerror', quantile)
            all_hpo_params[quantile_key] = params
            with open(hpo_file_path, 'w') as f: json.dump(all_hpo_params, f, indent=4)
            print(f"Saved new best params for {quantile_key} to {HPO_RESULTS_FILE}")
    if not params:
        params = {'learning_rate': 0.05, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9}
    params.update({'n_estimators': 2000, 'random_state': 42, 'objective': 'reg:quantileerror', 
                   'quantile_alpha': quantile, 'early_stopping_rounds': 50})
    main_model = XGBRegressor(**params)
    main_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    residual_model = None
    if fit_residuals and quantile == 0.50:
        print(f"Fitting residual model for P50...")
        main_preds_val = main_model.predict(X_val)
        residuals = y_val - main_preds_val
        X_train_res, y_train_res = X_val, residuals
        res_params = {'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 3, 'random_state': 42,
                      'objective': 'reg:squarederror', 'early_stopping_rounds': 20}
        residual_model = XGBRegressor(**res_params)
        residual_model.fit(X_train_res.iloc[:-10], y_train_res.iloc[:-10], 
                           eval_set=[(X_train_res.tail(10), y_train_res.tail(10))], verbose=False)
    return main_model, residual_model
def get_backtest_period(df):
    latest_date_with_data = df.index.max().date()
    for i in range(2, 10):
        test_date = latest_date_with_data - timedelta(days=i)
        test_start, test_end = pd.to_datetime(f"{test_date} 12:00:00"), pd.to_datetime(f"{test_date} 23:45:00")
        if (df.index >= test_start).sum() > 0 and (df.index <= test_end).sum() > 0:
            test_mask = (df.index >= test_start) & (df.index <= test_end)
            if test_mask.sum() == N_PREDICT:
                print(f"Found suitable backtest period: {test_date}")
                return test_date, test_mask
    raise ValueError("Could not find a suitable, contiguous 12-hour block for backtesting.")
def prepare_forecast_data(df, forecast_start, n_periods):
    forecast_end = forecast_start + timedelta(minutes=15 * n_periods)
    forecast_mask = (df['ts_start'] >= forecast_start) & (df['ts_start'] < forecast_end)
    forecast_data = df[forecast_mask].copy()
    if len(forecast_data) == 0: return None
    if forecast_data.isnull().values.any(): forecast_data.fillna(0, inplace=True)
    return forecast_data

if __name__ == "__main__":
    try: SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError: SCRIPT_DIR = os.getcwd()
    PATH = os.path.join(SCRIPT_DIR, "q_hourly_db_cleaned.csv")
    if not os.path.exists(PATH): raise FileNotFoundError(f"CSV file not found at: {PATH}")
    
    df_all, df_training = load_and_prepare(PATH)
    model_target = 'target_delta' if MODELING_STRATEGY == 'delta' else TARGET_COLUMN
    
    # --- Feature list creation ---
    feature_list = NUMERIC_FEATURES.copy()
    feature_list.extend([f for f in df_all.columns if 'minus' in f or 'ratio' in f])
    if USE_ROLLING_FEATURES: feature_list.extend([f for f in df_all.columns if "_roll_" in f])
    if USE_LAG_FEATURES: feature_list.extend([f for f in df_all.columns if "_lag" in f])
    if USE_DATE_FEATURES: feature_list.extend(['hour','dayofweek','is_weekend','hour_sin','hour_cos','time_in_session'])
    available_features = sorted(list(set(feature_list) & set(df_training.columns)))
    for col in [TARGET_COLUMN, 'target_delta', 'ts_start', 'ts_end']:
        if col in available_features: available_features.remove(col)
    print(f"\nUsing {len(available_features)} features to predict '{model_target}'.")

    # --- Main Execution Loop (Unchanged) ---
    for stage in ["BACKTESTING", "FORECASTING"]:
        print("\n" + "="*60 + f"\n{stage}\n" + "="*60)
        results = {}
        models = {}
        try:
            if stage == "BACKTESTING":
                backtest_date, backtest_mask = get_backtest_period(df_training)
                train_df, eval_df = df_training[~backtest_mask], df_training[backtest_mask]
            else: # FORECASTING
                train_df = df_training
                forecast_start = pd.to_datetime(f"{FORECAST_DATE} 12:00:00")
                eval_df = prepare_forecast_data(df_all, forecast_start, N_PREDICT)
                if eval_df is None: print("No forecast data available."); continue
            
            # Train models for all specified quantiles
            for q in QUANTILES:
                fit_residuals = (ENABLE_RESIDUAL_FITTING and q == 0.50)
                main_model, res_model = train_quantile_model(train_df, available_features, model_target, q, use_hpo=ENABLE_HPO, fit_residuals=fit_residuals)
                models[f'p{int(q*100)}'] = (main_model, res_model)
            
            # Generate predictions
            print(f"\n🔮 Generating {stage} predictions...")
            predictions = {}
            for q in QUANTILES:
                name = f'p{int(q*100)}'
                main_model, res_model = models[name]
                main_pred_deltas = main_model.predict(eval_df[available_features])
                if res_model: main_pred_deltas += res_model.predict(eval_df[available_features])
                predictions[name] = eval_df[DELTA_BASE_COLUMN].values + main_pred_deltas if MODELING_STRATEGY == 'delta' else main_pred_deltas
            
            if stage == "BACKTESTING":
                actuals = eval_df[TARGET_COLUMN].values
                mae = mean_absolute_error(actuals, predictions['p50'])
                smape_val = smape(actuals, predictions['p50'])
                print(f"\n🎯 Backtest Results for {backtest_date} (P50): MAE={mae:.3f}, sMAPE={smape_val:.2f}%")
                backtest_results = {"preds": predictions, "actuals": actuals, "data": eval_df, "date": backtest_date, "mae": mae, "smape": smape_val}
            else: # FORECASTING
                results_df = pd.DataFrame(predictions); results_df['timestamp'] = eval_df['ts_start'].values
                print("\nForecast Results (Final IDA3 Price Intervals):")
                print(results_df[['timestamp'] + [f'p{int(q*100)}' for q in QUANTILES]])
                forecast_results = {"preds": predictions, "data": eval_df}
        except Exception as e:
            print(f"❌ {stage} failed: {e}")

    # --- MODIFIED: Visualization with Layered Uncertainty Bands ---
    print("\n" + "="*60 + "\nGENERATING PLOTS\n" + "="*60)
    fig, axes = plt.subplots(2, 1, figsize=(16, 14), sharex=False)
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: Backtesting Results
    if backtest_results:
        br = backtest_results
        # Lighter red band for the full 90% interval (P5 to P95)
        axes[0].fill_between(br['data'].index, br['preds']['p5'], br['preds']['p95'], color='red', alpha=0.15, label='5%-95% Interval')
        # Darker red band for the 50% interquartile range (P25 to P75)
        axes[0].fill_between(br['data'].index, br['preds']['p25'], br['preds']['p75'], color='red', alpha=0.35, label='25%-75% Interval')
        axes[0].plot(br['data'].index, br['actuals'], label="Actual IDA3", color='blue', linewidth=2, zorder=10)
        axes[0].plot(br['data'].index, br['preds']['p50'], label="Predicted IDA3 (P50)", color='black', linestyle='--', linewidth=2, zorder=10)
        axes[0].set_title(f"Backtesting Results - {br['date']}\nMedian MAE: {br['mae']:.3f}, sMAPE: {br['smape']:.2f}%", fontsize=14)
        axes[0].set_ylabel("Price"); axes[0].legend()
    else: axes[0].set_title("Backtesting Results")

    # Plot 2: Forecast Results
    if forecast_results:
        fr = forecast_results
        # Lighter red band for the full 90% interval (P5 to P95)
        axes[1].fill_between(fr['data'].index, fr['preds']['p5'], fr['preds']['p95'], color='red', alpha=0.15, label='5%-95% Interval')
        # Darker red band for the 50% interquartile range (P25 to P75)
        axes[1].fill_between(fr['data'].index, fr['preds']['p25'], fr['preds']['p75'], color='red', alpha=0.35, label='25%-75% Interval')
        axes[1].plot(fr['data'].index, fr['preds']['p50'], label="Predicted IDA3 (P50)", color='black', linestyle='--', marker='o', markersize=4, zorder=10)
        last_day = df_training.tail(48)
        axes[1].plot(last_day.index, last_day[TARGET_COLUMN], label=f"Previous Day IDA3", color='gray', alpha=0.8)
        axes[1].set_title(f"Forecast Results - {FORECAST_DATE}", fontsize=14)
        axes[1].set_ylabel("Price"); axes[1].legend()
        axes[1].set_xlim(fr['data'].index.min(), fr['data'].index.max())
    else: axes[1].set_title(f"Forecast Results - {FORECAST_DATE}")
    
    fig.tight_layout(pad=3.0)
    output_filename = "ida3_forecast_and_backtest.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Chart saved to file: {output_filename}")