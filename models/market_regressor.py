import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb

CURRENT_YEAR = 2026

# Representative Indian Market Launch MSRP Map
MSRP_MAP = {
    ('Maruti', 'Swift'): 750000.0,
    ('Maruti', 'Baleno'): 850000.0,
    ('Maruti', 'Wagonr'): 650000.0,
    ('Maruti', 'Alto'): 450000.0,
    ('Maruti', 'Brezza'): 1150000.0,
    ('Maruti', 'Dzire'): 800000.0,
    ('Hyundai', 'Creta'): 1550000.0,
    ('Hyundai', 'I20'): 900000.0,
    ('Hyundai', 'Venue'): 1100000.0,
    ('Hyundai', 'Verna'): 1400000.0,
    ('Hyundai', 'Grand'): 700000.0,
    ('Honda', 'City'): 1450000.0,
    ('Honda', 'Amaze'): 850000.0,
    ('Tata', 'Nexon'): 1200000.0,
    ('Tata', 'Harrier'): 1950000.0,
    ('Tata', 'Tiago'): 700000.0,
    ('Tata', 'Safari'): 2200000.0,
    ('Mahindra', 'Scorpio'): 1600000.0,
    ('Mahindra', 'Xuv500'): 1700000.0,
    ('Mahindra', 'Xuv700'): 2100000.0,
    ('Mahindra', 'Thar'): 1650000.0,
    ('Toyota', 'Innova'): 2400000.0,
    ('Toyota', 'Fortuner'): 4000000.0,
    ('Toyota', 'Glanza'): 880000.0,
    ('Kia', 'Seltos'): 1600000.0,
    ('Kia', 'Sonet'): 1150000.0,
    ('Volkswagen', 'Polo'): 850000.0,
    ('Volkswagen', 'Taigun'): 1500000.0,
    ('Skoda', 'Slavia'): 1400000.0,
    ('Skoda', 'Kushaq'): 1500000.0,
    ('Bmw', '3 Series'): 4800000.0,
    ('Mercedes-benz', 'C-class'): 5500000.0,
    ('Audi', 'A4'): 4600000.0
}

TIER_1 = ['Maruti', 'Toyota', 'Honda', 'Hyundai']
TIER_2 = ['Tata', 'Mahindra', 'Kia', 'Volkswagen', 'Skoda', 'Ford', 'Renault', 'Nissan', 'Mg']
TIER_3 = ['Bmw', 'Mercedes-benz', 'Audi', 'Volvo', 'Jaguar', 'Porsche', 'Land rover']

NUM_COLS = ['vehicle_age', 'km_driven', 'km_per_year', 'engine_cc', 'max_power_bhp', 'power_to_displacement_ratio']
CAT_COLS = ['brand', 'fuel_type', 'transmission', 'brand_tier']
ALL_FEATURE_COLS = CAT_COLS + NUM_COLS

def get_brand_tier(brand: str) -> str:
    b = str(brand).capitalize()
    if b in TIER_1:
        return 'Tier_1_HighRetention'
    elif b in TIER_2:
        return 'Tier_2_Mainstream'
    elif b in TIER_3:
        return 'Tier_3_Luxury'
    return 'Tier_4_Other'

def get_launch_msrp(brand: str, model: str) -> float:
    key = (str(brand).capitalize(), str(model).capitalize())
    if key in MSRP_MAP:
        return MSRP_MAP[key]
    b = str(brand).capitalize()
    if b in TIER_3:
        return 5500000.0
    if b in ['Toyota', 'Mahindra', 'Jeep', 'Tata']:
        return 1800000.0
    return 950000.0

def engineer_features(specs: dict) -> pd.DataFrame:
    """
    Transforms raw user input dictionary into the exact tabular schema.
    """
    brand = str(specs.get('brand', 'Hyundai')).capitalize()
    model = str(specs.get('model', 'Creta')).capitalize()
    model_year = int(specs.get('model_year', 2020))
    km_driven = float(specs.get('km_driven', 45000))
    fuel_type = str(specs.get('fuel_type', 'Petrol')).capitalize()
    transmission = str(specs.get('transmission', 'Manual')).capitalize()
    engine_cc = float(specs.get('engine_cc', 1497.0))
    max_power_bhp = float(specs.get('max_power_bhp', 113.0))

    vehicle_age = max(0, CURRENT_YEAR - model_year)
    km_per_year = km_driven / (vehicle_age + 1.0)
    power_to_disp = max_power_bhp / (engine_cc + 1e-5)
    brand_tier = get_brand_tier(brand)

    row = {
        'brand': brand,
        'fuel_type': fuel_type,
        'transmission': transmission,
        'brand_tier': brand_tier,
        'vehicle_age': vehicle_age,
        'km_driven': km_driven,
        'km_per_year': km_per_year,
        'engine_cc': engine_cc,
        'max_power_bhp': max_power_bhp,
        'power_to_displacement_ratio': power_to_disp
    }
    return pd.DataFrame([row])

class MarketValuationEngine:
    def __init__(self, model_path: str = 'market_model_pipeline.joblib'):
        self.model_path = model_path
        self.pipeline = None
        self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                return
            except Exception as e:
                print(f"Warning: could not load {self.model_path}: {e}")

        # Train pipeline if dataset exists
        dataset_path = 'indian_market_master.csv'
        if os.path.exists(dataset_path):
            print(f"Training Market XGBoost Regressor on '{dataset_path}'...")
            df = pd.read_csv(dataset_path)
            X = df[ALL_FEATURE_COLS]
            y = df['depreciation_ratio']

            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), NUM_COLS),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS)
            ])

            self.pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', xgb.XGBRegressor(
                    n_estimators=450,
                    learning_rate=0.03,
                    max_depth=5,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    tree_method='hist',
                    random_state=42
                ))
            ])
            self.pipeline.fit(X, y)
            joblib.dump(self.pipeline, self.model_path)
            print(f"Saved trained market model pipeline to '{self.model_path}'")
        else:
            print("Notice: 'indian_market_master.csv' not found. Will use empirical fallback regressor.")

    def predict(self, specs: dict) -> dict:
        """
        Runs tabular inference and returns baseline Fair Market Value (FMV).
        """
        df_input = engineer_features(specs)
        brand = specs.get('brand', 'Hyundai')
        model = specs.get('model', 'Creta')
        custom_msrp = specs.get('launch_msrp', None)
        msrp = float(custom_msrp) if custom_msrp else get_launch_msrp(brand, model)

        if self.pipeline is not None:
            raw_ratio = float(self.pipeline.predict(df_input)[0])
        else:
            # High-fidelity empirical depreciation formula fallback
            age = df_input['vehicle_age'].values[0]
            km = df_input['km_driven'].values[0]
            tier = df_input['brand_tier'].values[0]
            tier_retention = 0.90 if tier == 'Tier_1_HighRetention' else (0.84 if tier == 'Tier_2_Mainstream' else 0.76)
            annual_dep = 0.08
            km_dep = (km / 100000.0) * 0.10
            raw_ratio = max(0.12, (tier_retention * ((1.0 - annual_dep) ** age)) - km_dep)

        # Guardrails: clamp depreciation ratio to realistic bounds [0.10, 1.05]
        depreciation_ratio = float(np.clip(raw_ratio, 0.10, 1.05))
        baseline_price = float(depreciation_ratio * msrp)

        return {
            'depreciation_ratio': depreciation_ratio,
            'launch_msrp': msrp,
            'baseline_price_inr': baseline_price,
            'features': df_input.iloc[0].to_dict()
        }
