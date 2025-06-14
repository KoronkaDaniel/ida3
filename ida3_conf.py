# ida3_conf.py

# Forecast target
TARGET_COLUMN = "ida3"

# Date of forecast (2025-06-13 12:00 to 00:00)
FORECAST_DATE = "2025-06-13"
FORECAST_HOURS = 12
FREQ = "15T"
N_PREDICT = 48

# We ONLY include features that are KNOWN for the future forecast period.
NUMERIC_FEATURES = [
    "dam", "idcth", "idcth_liq", "idctqh", "idctqh_liq",
    "idc3h", "idc3h_vol", "idc3qh", "idc3qh_vol",
    "ida1", "ida1_vol", "ida2", "ida2_vol",
    "temp_1", "irradiation_1",
    "temp_2", "irradiation_2",
    "net_load_prog"
]

# --- CORRECTED: Reduced lag/window sizes for small dataset ---
LAG_FEATURES = ["ida3", "ida3_vol", "net_load", "dam", "ida1", "ida2"]
# Max lag is now 24 steps (6 hours) instead of 48
LAG_STEPS = [1, 2, 4, 8, 12, 16, 24] 

USE_ROLLING_FEATURES = True
ROLLING_FEATURES_COLS = ['ida1', 'ida2', 'net_load', 'dam']
# Max window is now 24 steps (6 hours) instead of 48
ROLLING_WINDOW_SIZES = [4, 8, 12, 24]

USE_DATE_FEATURES = True
ENABLE_HPO = False