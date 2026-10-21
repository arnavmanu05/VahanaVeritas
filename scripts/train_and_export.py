import os
import sys

# Configure UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb

from models.market_regressor import NUM_COLS, CAT_COLS, ALL_FEATURE_COLS

def train_and_export_market_model():
    print("=" * 70)
    print("TRAINING BRANCH 1: MARKET REGRESSOR (XGBOOST)")
    print("=" * 70)
    dataset_path = 'indian_market_master.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: '{dataset_path}' not found.")
        return

    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df):,} records from '{dataset_path}'")

    X = df[ALL_FEATURE_COLS]
    y = df['depreciation_ratio']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUM_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method='hist',
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # Save artifact first
    out_file = 'market_model_pipeline.joblib'
    joblib.dump(pipeline, out_file)
    print(f"Exported pipeline artifact to '{out_file}'")

    y_pred_ratio = pipeline.predict(X_test)
    test_msrp = df.loc[X_test.index, 'launch_msrp'].values
    actual_inr = y_test.values * test_msrp
    pred_inr = y_pred_ratio * test_msrp

    r2 = r2_score(actual_inr, pred_inr)
    mae = mean_absolute_error(actual_inr, pred_inr)
    rmse = np.sqrt(mean_squared_error(actual_inr, pred_inr))
    mape = mean_absolute_percentage_error(actual_inr, pred_inr) * 100.0

    print(f"Evaluation on {len(X_test):,} held-out test vehicles:")
    print(f"  R2 Score:  {r2:.4f}")
    print(f"  MAE (INR): Rs. {mae:,.2f}")
    print(f"  RMSE (INR):Rs. {rmse:,.2f}")
    print(f"  MAPE:      {mape:.2f}%")
    print("=" * 70)

if __name__ == '__main__':
    train_and_export_market_model()
