import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from models.mechanical_lstm import ACTIVE_SENSOR_NAMES, generate_telemetry_preset

def export_all_presets():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sample_telemetry'))
    os.makedirs(out_dir, exist_ok=True)
    archetypes = ['pristine', 'nominal', 'neutral', 'neglected', 'lemon']
    for arch in archetypes:
        arr = generate_telemetry_preset(arch)
        df = pd.DataFrame(arr, columns=ACTIVE_SENSOR_NAMES)
        df.insert(0, 'cycle', np.arange(1, 31))
        out_csv = os.path.join(out_dir, f'telemetry_{arch}.csv')
        df.to_csv(out_csv, index=False)
        print(f"Exported {out_csv} ({len(df)} cycles, {len(ACTIVE_SENSOR_NAMES)} channels)")

if __name__ == '__main__':
    export_all_presets()
