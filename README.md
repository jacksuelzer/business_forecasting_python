# Business Forecasting with Python

## Overview
In this project, I examined 16 annoynmized properties for a major hotel chain. The goal was to identify how to best forecast the 4-weeks ahead occupancy rate for each property based on:   
  1. Historical demand;
  2. attributes (property type, etc);
  3. Available on-the-book data; and
  4. Any exogenous features I can extract.

 For my analysis, I examined five categories of forecasting models:  
   1. **Simple baselines**, which included the Naive and SeasonalNaive; 
   1. **Error trend seasonality**, where I examined the use of `AutoETS`;
   1. **Statistical models**, `AutoARIMA`;
   1. **Machine Learning models**, `LightGBM`;
   1. **Deep Learning models**, `AutoNHITS` and `AutoNBEATS`  ; 
   1. **Transformer models**, `TimeGPT`.

## Python Notebook
The Python Notebook containing my code, results, and insights is available at: [Python Hotel Forecasting Demo](https://colab.research.google.com/drive/1HKo7VRzrUcmY0nbYEUL9FPqgOe9dcTdu?usp=sharing).

## Business Insights

This project compares a bunch of modern time series forecasting models to figure out which one actually works best in a fast-changing business setting. The main goal was to test different approaches—Deep Learning, traditional statistical models, and classic Machine Learning—to see which one gives the most accurate predictions for things like future demand, inventory needs, or other key metrics.

After running cross-validation and evaluating everything using metrics like MAE and RMSE, here’s what stood out:

Deep Learning Models Perform the Best

Finding: Models like TimeGPT, NHITS, and NBEATS consistently had the lowest errors. They handled complicated patterns in the data—like multiple seasonalities and sudden shifts—much better than the other methods.
Why It Matters: Using the top model (TimeGPT) in production would likely lead to better demand planning, fewer stockouts, less excess inventory, and overall smoother operations.

Statistical Models Are a Solid Backup Option

Finding: Models such as AutoETS and AutoARIMA were surprisingly strong. They’re interpretable and reliable, and in some cases, they weren’t far behind Deep Learning models—especially when the data had clear patterns.
Use Case: These models are perfect for situations where you need low latency or want a dependable fallback model if the main system goes down.

Machine Learning Models Need Better Features

Finding: The LightGBM model didn’t perform as well, mostly because it only used basic date features. Without richer inputs, it couldn’t keep up with the deep learning or statistical models.
What This Means: To make tree-based ML models competitive, the business needs to invest in better feature engineering—things like adding promotional schedules, economic indicators, competitor pricing, and holiday/event flags. ML models depend heavily on good features, so domain knowledge becomes essential.

Key Packages Used: pandas, numpy, matplotlib / seaborn / plotly, statsmodels, neuralforecast / lightgbm

## Future Work & Next Steps

To move this from a prototype to something that can run in production, here are the recommended next steps:

Error Analysis & Explainability: Look closely at the time series where all models struggled. This helps identify outliers, missing variables, or unexpected events that weren’t captured.

Uncertainty Estimates: Add prediction intervals (like 80% and 95%) so the business can understand the risk around each forecast—not just the point prediction.

I wish I could have also implemented the cross validation more fluidly throughout the model allowing me to better analyze each hotel uniquely to figure out what the best model would be for each hotel. 

### Cross Validation for Naive and Seasonal Naive
# ---------------------------------------
# Train/Test Split
# ---------------------------------------
train = hotels.query('ds < "2023-06-30"')

# ---------------------------------------
# Baseline Forecasting Models
# ---------------------------------------
models = [
    Naive(),
    SeasonalNaive(season_length=28, alias="monthly_seasonality"),
]

# ---------------------------------------
# Initialize StatsForecast
# ---------------------------------------
sf = StatsForecast(
    models=models,
    freq="D"
)

# ---------------------------------------
# Cross-Validation Setup
# ---------------------------------------
# Note:
# Baseline statistical models do NOT use external predictors.
# Therefore, we subset to only the required columns.
# If you later add models that *do* take exogenous variables,
# run CV separately for both groups (baseline vs. with-predictors).
# ---------------------------------------
cross_df = sf.cross_validation(
    h=30,
    df=train[["unique_id", "ds", "y"]],
    n_windows=5,        # number of CV windows
    step_size=30        # non-overlapping windows
)

display(cross_df)

### AutoETS
# ---------------------------------------
# Imports
# ---------------------------------------
from statsforecast import StatsForecast
from statsforecast.models import AutoETS
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------
# Data Preparation
# ---------------------------------------
# Ensure dataset is sorted by date
hotels = hotels.sort_values(by="ds").reset_index(drop=True)

# Forecast horizon
h = 28

# TimeSeriesSplit setup
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)
splits = list(tscv.split(hotels))

autoets_predictions_per_fold = []

# ---------------------------------------
# Cross-Validation Loop
# ---------------------------------------
for i, (train_idx, val_idx) in enumerate(splits):
    # Split into train/validation sets
    train_df = hotels.iloc[train_idx].copy()
    val_df = hotels.iloc[val_idx].copy()

    # StatsForecast requires only id, timestamp, and target
    train_sf = train_df[["unique_id", "ds", "y"]].copy()

    # ---------------------------------------
    # Fit AutoETS Model
    # ---------------------------------------
    sf = StatsForecast(
        models=[AutoETS()],
        freq="D",
        n_jobs=-1
    )
    sf.fit(train_sf)

    # Forecast next h periods
    forecast_df = sf.predict(h=h)
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    # ---------------------------------------
    # Align Predictions with Validation Data
    # ---------------------------------------
    y_true_list = []
    y_pred_list = []

    for uid in forecast_df["unique_id"].unique():
        uid_pred = forecast_df[forecast_df["unique_id"] == uid].sort_values("ds")
        uid_actual = val_df[val_df["unique_id"] == uid].sort_values("ds").head(h)

        if not uid_pred.empty and not uid_actual.empty:
            merged = pd.merge(
                uid_actual[["unique_id", "ds", "y"]],
                uid_pred[["unique_id", "ds", "AutoETS"]],
                on=["unique_id", "ds"],
                how="inner"
            )
            y_true_list.extend(merged["y"].tolist())
            y_pred_list.extend(merged["AutoETS"].tolist())

    # Store fold results
    autoets_predictions_per_fold.append({
        "fold": i + 1,
        "y_true": y_true_list,
        "y_pred_autoets": y_pred_list,
        "predictions_df": merged
    })

# ---------------------------------------
# Output Summary
# ---------------------------------------
print(f"Generated AutoETS predictions for {len(autoets_predictions_per_fold)} folds.")
print(f"""
Example for Fold 1 (first 5 actual vs predicted):
Actual:    {autoets_predictions_per_fold[0]['y_true'][:5]}
Predicted: {autoets_predictions_per_fold[0]['y_pred_autoets'][:5]}
""")

### AutoAMIRA
# ---------------------------------------
# Imports
# ---------------------------------------
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import pandas as pd

# ---------------------------------------
# Forecast Horizon
# ---------------------------------------
h = 28   # predict next 28 days

autoarima_predictions_per_fold = []

# Note:
# `tscv_splits` must already be defined from previous TimeSeriesSplit setup.
# It should come from the AutoETS section.
# ---------------------------------------

# ---------------------------------------
# Cross-Validation Loop
# ---------------------------------------
for i, (train_idx, val_idx) in enumerate(tscv_splits):
    # Train and validation subsets
    train_df = hotels.iloc[train_idx].copy()
    val_df = hotels.iloc[val_idx].copy()

    # StatsForecast requires (unique_id, ds, y)
    train_sf = train_df[["unique_id", "ds", "y"]].copy()

    # ---------------------------------------
    # Fit AutoARIMA
    # ---------------------------------------
    sf = StatsForecast(
        models=[AutoARIMA()],
        freq="D",
        n_jobs=-1
    )
    sf.fit(train_sf)

    # Forecast next h periods
    forecast_df = sf.predict(h=h)
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    # ---------------------------------------
    # Align Predictions with Actuals
    # ---------------------------------------
    y_true_list = []
    y_pred_list = []

    for uid in forecast_df["unique_id"].unique():
        uid_pred = forecast_df[forecast_df["unique_id"] == uid].sort_values("ds")
        uid_actual = val_df[val_df["unique_id"] == uid].sort_values("ds").head(h)

        if not uid_pred.empty and not uid_actual.empty:
            merged = pd.merge(
                uid_actual[["unique_id", "ds", "y"]],
                uid_pred[["unique_id", "ds", "AutoARIMA"]],
                on=["unique_id", "ds"],
                how="inner"
            )
            y_true_list.extend(merged["y"].tolist())
            y_pred_list.extend(merged["AutoARIMA"].tolist())

    autoarima_predictions_per_fold.append({
        "fold": i + 1,
        "y_true": y_true_list,
        "y_pred_autoarima": y_pred_list,
        "predictions_df": merged    # Store merged predictions for metrics/analysis
    })

# ---------------------------------------
# Output Summary
# ---------------------------------------
print(f"Generated AutoARIMA predictions for {len(autoarima_predictions_per_fold)} folds.")
print(f"""
Example for Fold 1 (first 5 actual vs predicted):
Actual:    {autoarima_predictions_per_fold[0]['y_true'][:5]}
Predicted: {autoarima_predictions_per_fold[0]['y_pred_autoarima'][:5]}
""")

### LightGBM
# ---------------------------------------
# Imports
# ---------------------------------------
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np

# ---------------------------------------
# Forecast Horizon
# ---------------------------------------
h = 28   # predict next 28 days

lgbm_predictions_per_fold = []

# Date features automatically extracted by MLForecast
date_features = ['dayofweek', 'dayofyear', 'week', 'month', 'year']

# ---------------------------------------
# Cross-Validation Loop
# ---------------------------------------
for i, (train_idx, val_idx) in enumerate(tscv_splits):
    
    # Train and validation splits from expanded exogenous dataset
    train_df = hotels_exog.iloc[train_idx].copy()
    val_df = hotels_exog.iloc[val_idx].copy()

    # ---------------------------------------
    # Prepare Training Data for MLForecast
    # ---------------------------------------
    train_mlf = train_df[['unique_id_numeric', 'ds', 'y'] + exogenous_features].copy()
    train_mlf = train_mlf.rename(columns={'unique_id_numeric': 'unique_id'})

    # ---------------------------------------
    # Initialize MLForecast (LGBM)
    # ---------------------------------------
    model_mlf = MLForecast(
        models=[LGBMRegressor(random_state=42)],
        freq='D',
        date_features=date_features,
        lag_transforms={},         # no custom lag transforms
        target_transforms=[],      
        num_threads=-1
    )

    # Fit the model on the training data
    model_mlf.fit(train_mlf, static_features=[])

    # ---------------------------------------
    # Build Future DataFrame for Forecast Horizon
    # ---------------------------------------
    future_df = model_mlf.make_future_dataframe(h)

    # Prepare exogenous features for the validation portion
    val_exog = val_df[['unique_id_numeric', 'ds'] + exogenous_features].copy()
    val_exog = val_exog.rename(columns={'unique_id_numeric': 'unique_id'})

    # Merge future template with exogenous validation data
    X_df_predict = pd.merge(
        future_df,
        val_exog,
        on=['unique_id', 'ds'],
        how='left'
    )

    # Fill missing exogenous feature values (common for one-hot & OTB variables)
    X_df_predict[exogenous_features] = X_df_predict[exogenous_features].fillna(0)

    # Sort for MLForecast internal expectations
    X_df_predict = X_df_predict.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    # ---------------------------------------
    # Forecast Using MLForecast
    # ---------------------------------------
    forecast_df = model_mlf.predict(h=h, X_df=X_df_predict)
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

    # ---------------------------------------
    # Match Predictions with Actuals
    # ---------------------------------------
    y_true_list = []
    y_pred_list = []

    for uid in forecast_df['unique_id'].unique():
        uid_forecast = forecast_df[forecast_df['unique_id'] == uid].sort_values('ds')

        # Match validation actuals to forecasted dates
        uid_val = val_df[val_df['unique_id_numeric'] == uid].sort_values('ds')

        merged = pd.merge(
            uid_val[['unique_id_numeric', 'ds', 'y']].rename(columns={'unique_id_numeric': 'unique_id'}),
            uid_forecast[['unique_id', 'ds', 'LGBMRegressor']],
            on=['unique_id', 'ds'],
            how='inner'
        )

        if not merged.empty:
            y_true_list.extend(merged['y'].tolist())
            y_pred_list.extend(merged['LGBMRegressor'].tolist())

    # Store fold results
    lgbm_predictions_per_fold.append({
        "fold": i + 1,
        "y_true": y_true_list,
        "y_pred_lgbm": y_pred_list
    })

# ---------------------------------------
# Output Summary
# ---------------------------------------
print(f"Generated LGBM predictions for {len(lgbm_predictions_per_fold)} folds.")

if lgbm_predictions_per_fold:
    print(f"""
Example for Fold 1 (first 5 actual vs predicted):
Actual:    {lgbm_predictions_per_fold[0]['y_true'][:5]}
Predicted: {lgbm_predictions_per_fold[0]['y_pred_lgbm'][:5]}
""")
else:
    print("No predictions generated for any fold.")

### AutoNBEATS
# ---------------------------------------
# Imports
# ---------------------------------------
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
import pandas as pd

# ---------------------------------------
# Forecast Horizon
# ---------------------------------------
h = 28   # predict next 28 days

nbeats_predictions_per_fold = []

# ---------------------------------------
# Cross-Validation Loop
# ---------------------------------------
for i, (train_idx, val_idx) in enumerate(tscv_splits):
    
    # Train/validation splits from the NBEATS-ready dataset
    train_df = hotels_nf.iloc[train_idx].copy()
    val_df = hotels_nf.iloc[val_idx].copy()

    # ---------------------------------------
    # Initialize NBEATS Model
    # ---------------------------------------
    # input_size controls lookback window length
    # Common rule: input_size >= 2 * h
    nf = NeuralForecast(
        models=[
            NBEATS(
                h=h,
                input_size=2 * h,
                max_steps=100,
                learning_rate=1e-3,
                batch_size=32,
                random_seed=42,
                start_padding_enabled=True
            )
        ],
        freq='D'
    )

    # Fit the model (no internal validation used here)
    nf.fit(df=train_df)

    # ---------------------------------------
    # Forecast Next h Periods
    # ---------------------------------------
    forecast_df = nf.predict()
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

    # ---------------------------------------
    # Align Predictions with Validation Data
    # ---------------------------------------
    y_true_list = []
    y_pred_list = []

    for uid in forecast_df['unique_id'].unique():
        uid_pred = forecast_df[forecast_df['unique_id'] == uid].sort_values("ds")
        uid_actual = val_df[val_df['unique_id'] == uid].sort_values("ds").head(h)

        if not uid_pred.empty and not uid_actual.empty:
            merged = pd.merge(
                uid_actual[['unique_id', 'ds', 'y']],
                uid_pred[['unique_id', 'ds', 'NBEATS']],
                on=['unique_id', 'ds'],
                how='inner'
            )
            y_true_list.extend(merged['y'].tolist())
            y_pred_list.extend(merged['NBEATS'].tolist())

    # Store fold metrics
    nbeats_predictions_per_fold.append({
        "fold": i + 1,
        "y_true": y_true_list,
        "y_pred_nbeats": y_pred_list
    })

# ---------------------------------------
# Output Summary
# ---------------------------------------
print(f"Generated NBEATS predictions for {len(nbeats_predictions_per_fold)} folds.")
print(f"""
Example for Fold 1 (first 5 actual vs predicted):
Actual:    {nbeats_predictions_per_fold[0]['y_true'][:5]}
Predicted: {nbeats_predictions_per_fold[0]['y_pred_nbeats'][:5]}
""")


### AutoNHITS
# ---------------------------------------
# Imports
# ---------------------------------------
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
import pandas as pd

# ---------------------------------------
# Forecast Horizon
# ---------------------------------------
h = 28   # predict next 28 days

nhits_predictions_per_fold = []

# ---------------------------------------
# Cross-Validation Loop
# ---------------------------------------
for i, (train_idx, val_idx) in enumerate(tscv_splits):

    # Training and validation subsets
    train_df = hotels_nf.iloc[train_idx].copy()
    val_df = hotels_nf.iloc[val_idx].copy()

    # ---------------------------------------
    # Initialize NHITS Model
    # ---------------------------------------
    nf = NeuralForecast(
        models=[
            NHITS(
                h=h,
                input_size=2 * h,        # lookback window rule of thumb
                max_steps=100,
                learning_rate=1e-3,
                batch_size=32,
                random_seed=42,
                start_padding_enabled=True
            )
        ],
        freq='D'
    )

    # Fit the model
    nf.fit(df=train_df)

    # ---------------------------------------
    # Forecast Next h Periods
    # ---------------------------------------
    forecast_df = nf.predict()
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    # ---------------------------------------
    # Align Predictions with Actuals
    # ---------------------------------------
    y_true_list = []
    y_pred_list = []

    for uid in forecast_df["unique_id"].unique():
        uid_pred = forecast_df[forecast_df["unique_id"] == uid].sort_values("ds")
        uid_actual = val_df[val_df["unique_id"] == uid].sort_values("ds").head(h)

        if not uid_pred.empty and not uid_actual.empty:
            merged = pd.merge(
                uid_actual[["unique_id", "ds", "y"]],
                uid_pred[["unique_id", "ds", "NHITS"]],
                on=["unique_id", "ds"],
                how="inner"
            )
            y_true_list.extend(merged["y"].tolist())
            y_pred_list.extend(merged["NHITS"].tolist())

    # Store fold results
    nhits_predictions_per_fold.append({
        "fold": i + 1,
        "y_true": y_true_list,
        "y_pred_nhits": y_pred_list
    })

# ---------------------------------------
# Output Summary
# ---------------------------------------
print(f"Generated NHITS predictions for {len(nhits_predictions_per_fold)} folds.")
print(f"""
Example for Fold 1 (first 5 actual vs predicted):
Actual:    {nhits_predictions_per_fold[0]['y_true'][:5]}
Predicted: {nhits_predictions_per_fold[0]['y_pred_nhits'][:5]}
""")


### TimeGPT
# ---------------------------------------
# Imports
# ---------------------------------------
import pandas as pd
import numpy as np
from nixtla import NixtlaClient
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------------------
# TimeGPT Client Initialization
# ---------------------------------------
# nixtla_client = NixtlaClient(api_key="YOUR_API_KEY")
# Assumes `nixtla_client` is already authenticated in the environment.

# ---------------------------------------
# Cross-Validation Parameters
# ---------------------------------------
H_HORIZON = 7          # forecast horizon per fold (7-day backtest)
LAG_SIZE = 0           # no gap between train/validation
NUM_FOLDS = 5          # match 5-fold CV setup

print("Starting TimeGPT Cross-Validation to calculate RMSE, MAE, ME, and SMAPE...")

# ---------------------------------------
# Run TimeGPT Rolling Cross-Validation
# ---------------------------------------
timegpt_cv_results = nixtla_client.cross_validation(
    df=hotels_prepared,   # must contain: unique_id, ds (datetime), y
    h=H_HORIZON,
    n_windows=NUM_FOLDS,
    freq="D",
    id_col="unique_id",
    time_col="ds",
    target_col="y"
)

# ---------------------------------------
# SMAPE Calculation Function
# ---------------------------------------
def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Avoid divide-by-zero issues
    return np.mean(
        np.where(denominator == 0, 0, numerator / np.maximum(denominator, 1e-8))
    ) * 100


# ---------------------------------------
# Compute Metrics for Each Fold
# ---------------------------------------
metrics_per_fold = []

for cutoff, fold_df in timegpt_cv_results.groupby("cutoff"):
    y_true = fold_df["y"].values
    y_pred = fold_df["TimeGPT"].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    me = np.mean(y_pred - y_true)
    smape_val = smape(y_true, y_pred)

    metrics_per_fold.append({
        "fold": cutoff,
        "RMSE": rmse,
        "MAE": mae,
        "ME": me,
        "SMAPE": smape_val
    })


# ---------------------------------------
# Average Metrics Across Folds
# ---------------------------------------
avg_rmse = np.mean([m["RMSE"] for m in metrics_per_fold])
avg_mae = np.mean([m["MAE"] for m in metrics_per_fold])
avg_me = np.mean([m["ME"] for m in metrics_per_fold])
avg_smape = np.mean([m["SMAPE"] for m in metrics_per_fold])

# ---------------------------------------
# Output Summary
# ---------------------------------------
print("✅ TimeGPT Backtesting Complete.\n")
print("TimeGPT Performance Metrics (Averaged Across 5 Folds):")
print(f"Average RMSE:  {avg_rmse:.4f}")
print(f"Average MAE:   {avg_mae:.4f}")
print(f"Average ME:    {avg_me:.4f}")
print(f"Average SMAPE: {avg_smape:.4f}%")








