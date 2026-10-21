import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from transformers import AutoTokenizer, AutoModel

# ------------------------------------------------------------------------------
# 1. TRAIN BRANCH 2: MECHANICAL HEALTH BI-LSTM
# ------------------------------------------------------------------------------
def train_branch2_lstm(output_path="lstm_mhi_model.pth"):
    print("\n" + "=" * 70)
    print("TRAINING BRANCH 2: MECHANICAL HEALTH BI-LSTM (NASA C-MAPSS FD001)")
    print("=" * 70)

    filename = "train_FD001.txt"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"'{filename}' not found in root directory.")

    df = pd.read_csv(filename, sep=r"\s+", header=None, engine="python")
    max_c = df.groupby(0)[1].max().reset_index()
    max_c.columns = [0, "max_cycles"]
    df = pd.merge(df, max_c, on=0, how="left")
    df["mhi"] = (df["max_cycles"] - df[1]).clip(upper=125) / 125.0

    flat_sensors = [5, 9, 10, 14, 20, 22, 23] # invariant channels
    active_cols = [c for c in range(5, 26) if c not in flat_sensors]

    # Normalize active 14 channels
    for col in active_cols:
        c_min, c_max = df[col].min(), df[col].max()
        df[col] = (df[col] - c_min) / (c_max - c_min + 1e-8)

    WINDOW_SIZE = 30
    X_seq, y_seq = [], []
    for _, group in df.groupby(0):
        g_sensors = group[active_cols].values
        g_targets = group["mhi"].values
        n_steps = len(group)
        if n_steps >= WINDOW_SIZE:
            for i in range(n_steps - WINDOW_SIZE + 1):
                X_seq.append(g_sensors[i : i + WINDOW_SIZE])
                y_seq.append(g_targets[i + WINDOW_SIZE - 1])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    # 80/20 train/test split
    np.random.seed(42)
    indices = np.arange(len(X_seq))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, y_train = X_seq[train_idx], y_seq[train_idx]
    X_test, y_test = X_seq[test_idx], y_seq[test_idx]

    train_loader = DataLoader(
        list(zip(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))),
        batch_size=256,
        shuffle=True
    )
    test_loader = DataLoader(
        list(zip(torch.tensor(X_test), torch.tensor(y_test).unsqueeze(1))),
        batch_size=512,
        shuffle=False
    )

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

    model = MechanicalHealthLSTM(input_dim=14)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    EPOCHS = 8
    print(f"Training on {len(X_train):,} sequences for {EPOCHS} epochs...")
    model.train()
    start_t = time.time()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        avg_loss = total_loss / len(X_train)
        print(f"  Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.5f}")

    # Evaluation
    model.eval()
    test_preds, test_actuals = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            preds = model(bx).cpu().numpy().flatten()
            test_preds.extend(preds)
            test_actuals.extend(by.numpy().flatten())

    r2 = r2_score(test_actuals, test_preds)
    rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
    print(f"Bi-LSTM Evaluation: R²: {r2:.4f} | RMSE: {rmse:.4f} (Elapsed: {time.time()-start_t:.1f}s)")

    torch.save(model.state_dict(), output_path)
    print(f"✅ Successfully exported Bi-LSTM weights to '{output_path}'")

# ------------------------------------------------------------------------------
# 2. TRAIN BRANCH 3: MULTIMODAL DISTILBERT + MLP
# ------------------------------------------------------------------------------
def train_branch3_distilbert(output_path="distilbert_cs_model.pth"):
    print("\n" + "=" * 70)
    print("TRAINING BRANCH 3: MULTIMODAL DISTILBERT REGRESSOR (INSPECTION LOGS)")
    print("=" * 70)

    corpus_path = "synthetic_maintenance_corpus.csv"
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"'{corpus_path}' not found in root directory.")

    df_corpus = pd.read_csv(corpus_path)
    if df_corpus["fluid_condition"].dtype == object:
        cond_map = {"Good": 1.0, "Fair": 0.65, "Contaminated": 0.20}
        df_corpus["fluid_condition"] = df_corpus["fluid_condition"].map(cond_map).fillna(0.5)

    tab_cols = ["tire_tread_mm", "brake_pad_wear_pct", "fluid_condition", "service_adherence_ratio"]
    scaler = StandardScaler()
    df_corpus[tab_cols] = scaler.fit_transform(df_corpus[tab_cols])

    train_df, test_df = train_test_split(df_corpus, test_size=0.2, random_state=42)

    MODEL_NAME = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class MultimodalInspectionDataset(Dataset):
        def __init__(self, df, tok, max_len=64):
            self.texts = df["log_text"].tolist()
            self.tabular = df[tab_cols].values.astype(np.float32)
            self.targets = df["condition_score"].values.astype(np.float32)
            self.encodings = tok(
                self.texts,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "tabular": torch.tensor(self.tabular[idx]),
                "target": torch.tensor(self.targets[idx]).unsqueeze(0)
            }

    train_loader = DataLoader(MultimodalInspectionDataset(train_df, tokenizer), batch_size=32, shuffle=True)
    test_loader = DataLoader(MultimodalInspectionDataset(test_df, tokenizer), batch_size=64, shuffle=False)

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

    model_nlp = MultimodalConditionRegressor()
    optimizer = torch.optim.AdamW([
        {"params": model_nlp.transformer.parameters(), "lr": 3e-5},
        {"params": model_nlp.tab_mlp.parameters(), "lr": 1e-3},
        {"params": model_nlp.fusion_head.parameters(), "lr": 1e-3}
    ], weight_decay=0.01)
    criterion = nn.MSELoss()

    EPOCHS = 3
    print(f"Training on {len(train_df)} inspection records for {EPOCHS} epochs...")
    model_nlp.train()
    start_t = time.time()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            preds = model_nlp(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                tabular_features=batch["tabular"]
            )
            loss = criterion(preds, batch["target"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch["target"])
        avg_loss = total_loss / len(train_df)
        print(f"  Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.5f}")

    # Evaluation
    model_nlp.eval()
    test_preds, test_actuals = [], []
    with torch.no_grad():
        for batch in test_loader:
            preds = model_nlp(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                tabular_features=batch["tabular"]
            ).cpu().numpy().flatten()
            test_preds.extend(preds)
            test_actuals.extend(batch["target"].numpy().flatten())

    r2 = r2_score(test_actuals, test_preds)
    rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
    print(f"DistilBERT Evaluation: R²: {r2:.4f} | RMSE: {rmse:.4f} (Elapsed: {time.time()-start_t:.1f}s)")

    torch.save(model_nlp.state_dict(), output_path)
    print(f"✅ Successfully exported DistilBERT weights to '{output_path}'")

if __name__ == "__main__":
    if not os.path.exists("lstm_mhi_model.pth"):
        train_branch2_lstm("lstm_mhi_model.pth")
    else:
        print("Notice: 'lstm_mhi_model.pth' already exists. Skipping Bi-LSTM training.")
    train_branch3_distilbert("distilbert_cs_model.pth")
    print("\n" + "=" * 70)
    print("ALL NEURAL WEIGHTS GENERATED AND SAVED SUCCESSFULLY")
    print("=" * 70)
