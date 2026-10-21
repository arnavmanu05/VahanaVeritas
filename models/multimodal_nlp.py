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
                    max_length=64,
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
        Calculates token saliency using gradient x embedding or sentiment keyword scoring.
        """
        if getattr(self, 'is_fine_tuned', False) and self.model is not None and self.tokenizer is not None:
            try:
                import torch
                encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
                input_ids = encoded['input_ids']
                attention_mask = encoded['attention_mask']
                tab_norm = self._normalize_tabular(tab_dict)
                tab_tensor = torch.tensor(tab_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

                embeds = self.model.transformer.embeddings.word_embeddings(input_ids)
                embeds.retain_grad()

                output = self.model(inputs_embeds=embeds, attention_mask=attention_mask, tabular_features=tab_tensor)
                self.model.zero_grad()
                output.backward()

                grads = embeds.grad[0].cpu().numpy()
                raw_embeds = embeds[0].detach().cpu().numpy()
                saliency = np.sum(grads * raw_embeds, axis=1)
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

                valid_tokens = []
                valid_scores = []
                for t, s in zip(tokens, saliency):
                    if t not in ['[CLS]', '[SEP]', '[PAD]']:
                        valid_tokens.append(t)
                        valid_scores.append(float(s))
                return valid_tokens, valid_scores
            except Exception as e:
                print(f"Error computing neural saliency: {e}")

        # Fallback keyword importance dictionary
        words = text.split()
        scores = []
        lower_words = [w.lower().strip(".,!?:;") for w in words]
        severe = {'seized': -0.8, 'sludge': -0.7, 'blown': -0.9, 'cracked': -0.8, 'overheating': -0.7, 'starvation': -0.8}
        minor = {'noise': -0.3, 'sweating': -0.2, 'worn': -0.3, 'resurfaced': -0.2, 'fault': -0.4}
        pos = {'regular': 0.5, 'periodic': 0.4, 'passed': 0.6, 'nominal': 0.5, 'clean': 0.4, 'new': 0.5}

        for w in lower_words:
            if w in severe: scores.append(severe[w])
            elif w in minor: scores.append(minor[w])
            elif w in pos: scores.append(pos[w])
            else: scores.append(0.0)
        return words, scores
