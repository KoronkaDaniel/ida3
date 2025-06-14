# ida3_conf.py

# --- Modeling & Feature Strategy ---
MODELING_STRATEGY = 'delta'
DELTA_BASE_COLUMN = 'ida1'
TARGET_COLUMN = "ida3"

# --- NEW: Quantiles for Prediction Intervals ---
# Note: Corrected the duplicate 0.75
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

# --- Advanced Modeling Switches ---
ENABLE_HPO = False # Set to True to FIND and SAVE the best params.
ENABLE_RESIDUAL_FITTING = True 

# --- File to store HPO results ---
HPO_RESULTS_FILE = "best_hpo_params.json"

# --- Forecast Date ---
FORECAST_DATE = "2025-06-13"
N_PREDICT = 48

# --- Feature Engineering Lists ---
NUMERIC_FEATURES = [
    "dam", "idcth", "idcth_liq", "idctqh", "idctqh_liq",
    "idc3h", "idc3h_vol", "idc3qh", "idc3qh_vol",
    "ida1", "ida1_vol", "ida2", "ida2_vol",
    "temp_1", "irradiation_1",
    "temp_2", "irradiation_2",
    "net_load_prog"
]
USE_LAG_FEATURES = True
LAG_FEATURES = ["ida3", "ida3_vol", "net_load", "dam", "ida1", "ida2"]
LAG_STEPS = [1, 2, 4, 8, 12, 16, 24, 48] 
USE_ROLLING_FEATURES = True
ROLLING_FEATURES_COLS = ['ida1', 'ida2', 'net_load', 'dam']
ROLLING_WINDOW_SIZES = [4, 8, 16, 24, 48] 
USE_DATE_FEATURES = True