import os
import numpy as np
import pandas as pd

# Active 14 channels out of 21 C-MAPSS channels (dropping 1, 5, 6, 10, 16, 18, 19)
ACTIVE_SENSOR_NAMES = [
    "LPC Outlet Temp (S2)",
    "HPC Outlet Temp (S3)",
    "LPT Outlet Temp (S4)",
    "HPC Outlet Pressure (S7)",
    "Physical Fan Speed (S8)",
    "Physical Core Speed (S9)",
    "Static HPC Pressure (S11)",
    "Fuel-to-Air Ratio (S12)",
    "Corrected Fan Speed (S13)",
    "Corrected Core Speed (S14)",
    "Bypass Ratio (S15)",
    "Bleed Enthalpy (S17)",
    "HPT Coolant Bleed (S20)",
    "LPT Coolant Bleed (S21)"
]

WINDOW_SIZE = 30
INPUT_DIM = 14

class MechanicalHealthEngine:
    def __init__(self, model_path: str = 'lstm_mhi_model.pth'):
        self.model_path = model_path
        self.model = None
        self.device = "cpu"
        self._init_model()

    def _init_model(self):
        try:
            import torch
            import torch.nn as nn
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            class MechanicalHealthLSTM(nn.Module):
                def __init__(self, input_dim=14, hidden_dim=64, num_layers=2, dropout=0.2):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_dim,
                        hidden_size=hidden_dim,
                        num_layers=num_layers,
                        batch_first=True,
                        bidirectional=True,
                        dropout=dropout
                    )
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_dim * 2, 32),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(32, 1),
                        nn.Sigmoid()
                    )
                def forward(self, x):
                    out, _ = self.lstm(x)
                    last_hidden = out[:, -1, :]
                    return self.fc(last_hidden)

            self.model = MechanicalHealthLSTM(input_dim=INPUT_DIM).to(self.device)
            self.is_trained = False
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.is_trained = True
                print(f"Loaded Mechanical Health Bi-LSTM from '{self.model_path}'")
            else:
                print("Notice: 'lstm_mhi_model.pth' not found. Using calibrated sequence degradation estimator.")
            self.model.eval()
        except Exception as e:
            self.is_trained = False
            print(f"PyTorch or LSTM initialization notice: {e}. Fallback heuristics ready.")

    def predict(self, sequence_array: np.ndarray) -> dict:
        """
        Ingests a sequence array of shape (30, 14) or (batch, 30, 14)
        Returns:
            mhi: float in [0.0, 1.0]
            status: str
            channel_means: dict
        """
        arr = np.array(sequence_array, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0) # (1, 30, 14)

        # Ensure correct shape
        if arr.shape[1] != WINDOW_SIZE or arr.shape[2] != INPUT_DIM:
            # Resample or pad/truncate
            resampled = np.zeros((arr.shape[0], WINDOW_SIZE, INPUT_DIM), dtype=np.float32)
            for b in range(arr.shape[0]):
                for c in range(min(arr.shape[2], INPUT_DIM)):
                    col_data = arr[b, :, c]
                    resampled[b, :, c] = np.interp(
                        np.linspace(0, len(col_data) - 1, WINDOW_SIZE),
                        np.arange(len(col_data)),
                        col_data
                    )
            arr = resampled

        # Clip values to [0.0, 1.0]
        arr = np.clip(arr, 0.0, 1.0)

        # PyTorch model inference
        mhi_val = None
        if getattr(self, 'is_trained', False) and self.model is not None:
            try:
                import torch
                with torch.no_grad():
                    inp_t = torch.tensor(arr, dtype=torch.float32).to(self.device)
                    out = self.model(inp_t)
                    mhi_val = float(out.cpu().numpy().flatten()[0])
            except Exception as e:
                print(f"Inference error in LSTM: {e}")

        # Fallback heuristic calculation if model not yet loaded
        if mhi_val is None:
            # Degradation trend: earlier cycles vs recent cycles
            # Higher degradation typically leads to higher temperature/vibration drift towards 1.0
            recent_mean = float(np.mean(arr[0, -10:, :]))
            early_mean = float(np.mean(arr[0, :10, :]))
            drift = recent_mean - early_mean
            mhi_val = float(np.clip(1.0 - (recent_mean * 0.7 + max(0, drift) * 0.5), 0.05, 0.98))

        # Categorize health status
        if mhi_val >= 0.85:
            status = "Pristine"
            badge_color = "#10b981"
        elif mhi_val >= 0.70:
            status = "Well-Maintained"
            badge_color = "#0284c7"
        elif mhi_val >= 0.50:
            status = "Baseline"
            badge_color = "#d97706"
        elif mhi_val >= 0.30:
            status = "Minor Faults"
            badge_color = "#ea580c"
        else:
            status = "Critical Defects"
            badge_color = "#ef4444"

        channel_averages = {
            ACTIVE_SENSOR_NAMES[i]: float(np.mean(arr[0, :, i]))
            for i in range(INPUT_DIM)
        }

        return {
            'mechanical_health_index': float(mhi_val),
            'status': status,
            'badge_color': badge_color,
            'channel_averages': channel_averages,
            'sequence_tensor': arr[0]
        }

def generate_telemetry_preset(archetype: str) -> np.ndarray:
    """
    Synthesizes a realistic 30x14 telemetry sequence corresponding to vehicle wear archetypes.
    """
    np.random.seed(42 if archetype == 'pristine' else 123)
    time_steps = 30
    channels = 14

    if archetype == 'pristine':
        # Low noise, tight near baseline (0.15 - 0.25)
        base = np.linspace(0.18, 0.22, time_steps)[:, None]
        noise = np.random.normal(0, 0.02, (time_steps, channels))
        seq = base + noise
    elif archetype == 'nominal':
        # Moderate stable operation (0.35 - 0.45)
        base = np.linspace(0.32, 0.42, time_steps)[:, None]
        noise = np.random.normal(0, 0.03, (time_steps, channels))
        seq = base + noise
    elif archetype == 'neutral':
        # Average uninspected vehicle (0.50)
        base = np.linspace(0.48, 0.52, time_steps)[:, None]
        noise = np.random.normal(0, 0.04, (time_steps, channels))
        seq = base + noise
    elif archetype == 'neglected':
        # High thermal/pressure drift (0.60 - 0.75)
        base = np.linspace(0.55, 0.72, time_steps)[:, None]
        noise = np.random.normal(0, 0.05, (time_steps, channels))
        seq = base + noise
    elif archetype == 'lemon':
        # Extreme divergence / vibration / degradation (0.75 - 0.95)
        base = np.linspace(0.70, 0.95, time_steps)[:, None]
        noise = np.random.normal(0, 0.08, (time_steps, channels))
        seq = base + noise
    else:
        seq = np.full((time_steps, channels), 0.5)

    return np.clip(seq, 0.0, 1.0).astype(np.float32)
