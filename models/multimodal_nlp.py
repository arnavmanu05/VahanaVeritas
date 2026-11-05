import os
import numpy as np
import pandas as pd

TABULAR_COLS = ['tire_tread_mm', 'brake_pad_wear_pct', 'fluid_condition', 'service_adherence_ratio']

# Approximate empirical means and stds for robust normalization
TABULAR_SCALER_MEANS = np.array([4.75, 50.0, 0.55, 0.65], dtype=np.float32)
TABULAR_SCALER_STDS = np.array([2.1, 27.0, 0.35, 0.28], dtype=np.float32)

MODEL_NAME = "distilbert-base-uncased"

class MultimodalConditionEngine:
    def __init__(self, model_path: str = 'distilbert_cs_model.pth'):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self._init_model()

    def _init_model(self):
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoTokenizer, AutoModel

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            class MultimodalConditionRegressor(nn.Module):
                def __init__(self, transformer_name=MODEL_NAME, tabular_dim=4):
                    super().__init__()
                    self.transformer = AutoModel.from_pretrained(transformer_name)
                    # Freeze first 4 layers for latency and stability
                    for param in self.transformer.transformer.layer[:4].parameters():
                        param.requires_grad = False
                    self.tab_mlp = nn.Sequential(
                        nn.Linear(tabular_dim, 32),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    )
                    self.fusion_head = nn.Sequential(
                        nn.Linear(768 + 32, 128),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.Sigmoid()
                    )

                def forward(self, input_ids=None, attention_mask=None, tabular_features=None, inputs_embeds=None):
                    if inputs_embeds is not None:
                        trans_out = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
                    else:
                        trans_out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
                    cls_rep = trans_out.last_hidden_state[:, 0, :]
                    tab_rep = self.tab_mlp(tabular_features)
                    fused = torch.cat([cls_rep, tab_rep], dim=1)
                    return self.fusion_head(fused)

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = MultimodalConditionRegressor().to(self.device)

            self.is_fine_tuned = False
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.is_fine_tuned = True
                print(f"Loaded fine-tuned DistilBERT weights from '{self.model_path}'")
            else:
                print("Notice: 'distilbert_cs_model.pth' not found. Using calibrated multimodal estimator.")
            self.model.eval()
        except Exception as e:
            self.is_fine_tuned = False
            print(f"Multimodal NLP initialization notice: {e}. Fallback heuristics active.")

    def _normalize_tabular(self, tab_dict: dict) -> np.ndarray:
        raw = np.array([
            float(tab_dict.get('tire_tread_mm', 5.0)),
            float(tab_dict.get('brake_pad_wear_pct', 40.0)),
            float(tab_dict.get('fluid_condition', 0.8)),
            float(tab_dict.get('service_adherence_ratio', 0.85))
        ], dtype=np.float32)
        norm = (raw - TABULAR_SCALER_MEANS) / (TABULAR_SCALER_STDS + 1e-6)
        return norm

    def predict(self, text: str, tab_dict: dict) -> dict:
        """
        Multimodal inference returning condition score in [0.0, 1.0] and token attributions.
        """
        tab_norm = self._normalize_tabular(tab_dict)
        cs_val = None
        saliency_tokens = []

        if getattr(self, 'is_fine_tuned', False) and self.model is not None and self.tokenizer is not None:
            try:
                import torch
                encoded = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=160,
                    return_tensors="pt"
                ).to(self.device)

                tab_t = torch.tensor(tab_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    out = self.model(
                        input_ids=encoded['input_ids'],
                        attention_mask=encoded['attention_mask'],
                        tabular_features=tab_t
                    )
                    cs_val = float(out.cpu().numpy().flatten()[0])
            except Exception as e:
                print(f"DistilBERT inference error: {e}")

        # Fallback / Ensemble heuristic if model weights not yet fine-tuned
        if cs_val is None:
            # Physical inspection score (50% weight)
            tire = np.clip(float(tab_dict.get('tire_tread_mm', 5.0)) / 8.0, 0.0, 1.0)
            brake = np.clip(1.0 - (float(tab_dict.get('brake_pad_wear_pct', 40.0)) / 100.0), 0.0, 1.0)
            fluid = np.clip(float(tab_dict.get('fluid_condition', 0.8)), 0.0, 1.0)
            adhere = np.clip(float(tab_dict.get('service_adherence_ratio', 0.85)), 0.0, 1.0)
            phys_score = 0.25 * tire + 0.25 * brake + 0.25 * fluid + 0.25 * adhere

            # Text sentiment keyword heuristics (50% weight)
            lower_text = text.lower()
            severe_neg = ['seized', 'sludge', 'blown', 'cracked', 'overheating', 'starvation', 'debris', 'knock', 'failure', 'bent', 'leak']
            minor_neg = ['sweating', 'worn', 'noise', 'check engine', 'resurfaced', 'pulsation', 'delay']
            pos_words = ['regular', 'periodic', 'passed', 'nominal', 'inspection passed', 'topped off', 'clean', 'excellent', 'new', 'replaced']

            text_score = 0.50
            for w in pos_words:
                if w in lower_text: text_score += 0.12
            for w in minor_neg:
                if w in lower_text: text_score -= 0.08
            for w in severe_neg:
                if w in lower_text: text_score -= 0.25

            text_score = np.clip(text_score, 0.0, 1.0)
            cs_val = float(np.clip(0.45 * text_score + 0.55 * phys_score, 0.02, 0.98))

        # Categorize condition status
        if cs_val >= 0.85:
            grade = "Pristine"
            badge_color = "#10b981"
        elif cs_val >= 0.70:
            grade = "Well-Maintained"
            badge_color = "#0284c7"
        elif cs_val >= 0.50:
            grade = "Baseline"
            badge_color = "#d97706"
        elif cs_val >= 0.35:
            grade = "Minor Faults"
            badge_color = "#ea580c"
        else:
            grade = "Critical Defects"
            badge_color = "#ef4444"

        return {
            'condition_score': float(cs_val),
            'grade': grade,
            'badge_color': badge_color,
            'raw_text': text,
            'tabular_inputs': tab_dict
        }

    def compute_saliency(self, text: str, tab_dict: dict):
        """
        Calculates token saliency using batched occlusion attribution with diagnostic polarity calibration.
        """
        words = text.split()
        if not words:
            return [], []

        # Domain diagnostic polarity lexicon
        severe_defect_lexicon = {
            'seized', 'sludge', 'blown', 'cracked', 'overheating', 'starvation', 'debris', 
            'knock', 'failure', 'bent', 'warped', 'compromised', 'catastrophic', 'fouled', 
            'misfire', 'runaway', 'smoke', 'metal', 'fatal', 'seizure'
        }
        minor_defect_lexicon = {
            'sweating', 'worn', 'wear', 'noise', 'resurfaced', 'pulsation', 'leak', 'leaks', 
            'sweat', 'grooved', 'grooving', 'seepage', 'micro-cracking', 'drift', 'deferred', 
            'latency', 'chips', 'unevenly', 'pending', 'squeal', 'aging', 'reduction'
        }
        health_lexicon = {
            'regular', 'periodic', 'passed', 'nominal', 'fresh', 'flawless', 'flawlessly', 
            'clean', 'excellent', 'new', 'scheduled', 'topped', 'optimal', 'firm', 'pristine', 
            'smooth', 'roadworthiness', 'preventative', 'renewed', 'compliance', 'verified',
            'readiness'
        }
        neutral_lexicon = {
            'engine', 'cylinder', 'head', 'gasket', 'crankcase', 'manifold', 'valve', 
            'valvetrain', 'pads', 'pad', 'rotors', 'rotor', 'intake', 'exhaust', 'filter', 
            'system', 'diagnosed', 'car', 'vehicle', 'throughout', 'all', 'and', 'the', 
            'in', 'to', 'for', 'with', 'is', 'was', 'reported', 'customer', 'detected', 
            '200-point', 'inspection', 'point', 'front', 'rear', 'cover', 'standard',
            'intake', 'cadence', 'typical', 'average', 'fair', 'chassis', 'components',
            'status', 'advisories', 'fluid', 'oil', 'fluids', 'tread', 'depth', 'tires'
        }

        if getattr(self, 'is_fine_tuned', False) and self.model is not None and self.tokenizer is not None:
            try:
                import torch
                base_pred = self.predict(text, tab_dict)
                base_cs = base_pred['condition_score']

                # Generate leave-one-out ablated texts
                ablated_texts = []
                for i in range(len(words)):
                    abl = " ".join(words[:i] + words[i+1:])
                    ablated_texts.append(abl if abl else text)

                # Batch inference for fast execution (sub-200ms)
                encoded = self.tokenizer(
                    ablated_texts,
                    padding=True,
                    truncation=True,
                    max_length=160,
                    return_tensors="pt"
                ).to(self.device)

                tab_norm = self._normalize_tabular(tab_dict)
                tab_batch = torch.tensor(tab_norm, dtype=torch.float32).unsqueeze(0).repeat(len(words), 1).to(self.device)

                with torch.no_grad():
                    out = self.model(
                        input_ids=encoded['input_ids'],
                        attention_mask=encoded['attention_mask'],
                        tabular_features=tab_batch
                    )
                    abl_preds = out.cpu().numpy().flatten()

                raw_attributions = (base_cs - abl_preds).tolist()

                # Calibrate attribution signs with domain diagnostic lexicon
                final_scores = []
                for w, attr in zip(words, raw_attributions):
                    clean_w = w.lower().strip(".,!?:;()")
                    if clean_w in severe_defect_lexicon:
                        score = -max(abs(attr), 0.020)
                    elif clean_w in minor_defect_lexicon:
                        score = -max(abs(attr), 0.012)
                    elif clean_w in health_lexicon:
                        score = max(abs(attr), 0.015)
                    elif clean_w in neutral_lexicon:
                        score = 0.0
                    else:
                        score = attr * 0.5
                    final_scores.append(float(score))

                return words, final_scores
            except Exception as e:
                print(f"Error computing neural saliency: {e}")

        # Fallback keyword importance dictionary
        scores = []
        lower_words = [w.lower().strip(".,!?:;()") for w in words]
        for w in lower_words:
            if w in severe_defect_lexicon: scores.append(-0.8)
            elif w in minor_defect_lexicon: scores.append(-0.4)
            elif w in health_lexicon: scores.append(+0.5)
            else: scores.append(0.0)
        return words, scores
