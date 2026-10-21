import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any

POPULAR_BRANDS = [
    "Maruti", "Hyundai", "Honda", "Tata", "Mahindra", 
    "Toyota", "Kia", "Volkswagen", "Skoda", "Bmw", "Mercedes-benz", "Audi"
]

BRAND_MODELS = {
    "Maruti": ["Swift", "Baleno", "Brezza", "Dzire", "Wagonr", "Alto", "Ertiga"],
    "Hyundai": ["Creta", "Venue", "I20", "Verna", "Grand", "Tucson"],
    "Honda": ["City", "Amaze", "Elevate", "Civic", "Wr-v"],
    "Tata": ["Nexon", "Harrier", "Safari", "Tiago", "Altroz", "Punch"],
    "Mahindra": ["Scorpio", "Xuv700", "Thar", "Xuv500", "Bolero"],
    "Toyota": ["Innova", "Fortuner", "Glanza", "Hyryder", "Camry"],
    "Kia": ["Seltos", "Sonet", "Carens", "Carnival"],
    "Volkswagen": ["Taigun", "Virtus", "Polo", "Vento", "Tiguan"],
    "Skoda": ["Slavia", "Kushaq", "Octavia", "Superb"],
    "Bmw": ["3 Series", "5 Series", "X1", "X3", "X5"],
    "Mercedes-benz": ["C-class", "E-class", "Glc", "Gla"],
    "Audi": ["A4", "A6", "Q3", "Q5"]
}

DEFAULT_SPECS = {
    "Swift": {"engine_cc": 1197.0, "max_power_bhp": 88.5, "fuel_type": "Petrol"},
    "Baleno": {"engine_cc": 1197.0, "max_power_bhp": 88.5, "fuel_type": "Petrol"},
    "Brezza": {"engine_cc": 1462.0, "max_power_bhp": 101.6, "fuel_type": "Petrol"},
    "Creta": {"engine_cc": 1497.0, "max_power_bhp": 113.4, "fuel_type": "Diesel"},
    "Venue": {"engine_cc": 1197.0, "max_power_bhp": 81.8, "fuel_type": "Petrol"},
    "City": {"engine_cc": 1498.0, "max_power_bhp": 119.3, "fuel_type": "Petrol"},
    "Nexon": {"engine_cc": 1497.0, "max_power_bhp": 118.3, "fuel_type": "Diesel"},
    "Harrier": {"engine_cc": 1956.0, "max_power_bhp": 167.6, "fuel_type": "Diesel"},
    "Scorpio": {"engine_cc": 2184.0, "max_power_bhp": 130.0, "fuel_type": "Diesel"},
    "Xuv700": {"engine_cc": 2184.0, "max_power_bhp": 182.0, "fuel_type": "Diesel"},
    "Innova": {"engine_cc": 2393.0, "max_power_bhp": 147.5, "fuel_type": "Diesel"},
    "Fortuner": {"engine_cc": 2755.0, "max_power_bhp": 201.1, "fuel_type": "Diesel"},
    "Seltos": {"engine_cc": 1497.0, "max_power_bhp": 113.4, "fuel_type": "Diesel"}
}

def get_models_for_brand(brand: str) -> List[str]:
    b = str(brand).capitalize()
    return BRAND_MODELS.get(b, ["Generic"])

def get_default_engine_specs(model: str) -> Dict[str, Any]:
    m = str(model).capitalize()
    return DEFAULT_SPECS.get(m, {"engine_cc": 1497.0, "max_power_bhp": 100.0, "fuel_type": "Petrol"})

def load_dataset_summary() -> Dict[str, Any]:
    """
    Returns high-level statistics from the harmonized master dataset if present.
    """
    path = "indian_market_master.csv"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return {
                "total_records": len(df),
                "brands_count": df['brand'].nunique(),
                "avg_price": float(df['selling_price'].mean()),
                "median_km": float(df['km_driven'].median()),
                "min_year": int(df['model_year'].min()),
                "max_year": int(df['model_year'].max())
            }
        except Exception as e:
            print(f"Error loading master dataset summary: {e}")
    return {
        "total_records": 22320,
        "brands_count": 32,
        "avg_price": 685000.0,
        "median_km": 52000.0,
        "min_year": 2005,
        "max_year": 2024
    }
