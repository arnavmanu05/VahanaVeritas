# VahanaVeritas: Multimodal Used Vehicle Valuation & Condition Assessment Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Macro--Reg-orange.svg)](https://xgboost.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Bi--LSTM-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-DistilBERT-yellow.svg)](https://huggingface.co/)

---

## 1. Executive Summary & Problem Formulation

In emerging automotive markets such as India, the pre-owned vehicle ecosystem is severely plagued by **information asymmetry**—the classical economic phenomenon formalized in George Akerlof's *The Market for Lemons*. Traditional valuation algorithms rely almost exclusively on macro tabular indicators: vehicle age, odometer reading (kilometer count), fuel type, and brand.

Under conventional tabular-only pricing models, two identical vehicles with the same age and mileage receive identical valuations, regardless of internal mechanical integrity or deferred maintenance:
- **Vehicle A**: Highway-driven, strict OEM periodic service compliance, pristine mechanical compression, zero diagnostic error codes.
- **Vehicle B**: City-driven under thermal stress, delayed oil changes leading to valvetrain sludge, resurfaced brake pads, worn suspension bushings, or silent head-gasket failure.

Buyers face adverse selection and discount all offerings by an average **18% to 25% discount**, punishing conscientious sellers and incentivizing adverse selection.

**VahanaVeritas** resolves this fundamental market failure through a **Tri-Branch Multimodal Late-Fusion Architecture** that ingests:
1. **Macro Market Tabular Regressors** (Resale listings from CarDekho and Cars24).
2. **High-Frequency Sensor Telemetry Sequences** (Mechanical Health Index via Bi-Directional LSTM).
3. **Multimodal Diagnostic Records & Inspection Text** (Condition Score via DistilBERT + MLP Early Fusion).

---

## 2. System Architecture

The pipeline unifies three independent branches through a calibrated, interpretable **Late-Fusion Elasticity Engine**:

```
+-------------------------------------------------------------------------------+
|                             VAHANAVERITAS ENGINE                              |
+-------------------------------------------------------------------------------+
                                        |
       +--------------------------------+-------------------------------+
       |                                |                               |
[Branch 1: Market XGBoost]   [Branch 2: Bi-LSTM Telemetry]   [Branch 3: DistilBERT NLP]
       |                                |                               |
       v                                v                               v
 Baseline Market FMV (₹)         Mechanical Health               Physical Condition
     P_base in INR                   MHI in [0, 1]                   CS in [0, 1]
       |                                |                               |
       +--------------------------------+-------------------------------+
                                        |
                                        v
                       +---------------------------------+
                       |       LATE-FUSION ENGINE        |
                       | M = 1 + w1(MHI-0.5) + w2(CS-0.5)|
                       +---------------------------------+
                                        |
                                        v
                       Condition-Adjusted Resale Value (₹)
                           P_final = P_base * M
                                        |
                                        v
                       Explainability (SHAP + Token Saliency)
```

### 2.1 Branch 1: Tabular Market Valuation Regressor (XGBoost)
- **Objective**: Establish baseline Fair Market Value ($P_{\text{base}}$) capturing macroeconomic depreciation patterns across the Indian market.
- **Target Formulation**: Predicts the dimensionless **Depreciation Ratio** $y_{\text{ratio}} \in [0.10, 1.10]$ to prevent scale distortion across luxury vs. economy models:
  $$y_{\text{ratio}} = \frac{\text{Selling Price}}{\text{Launch MSRP}}$$
  $$P_{\text{base}} = \hat{y}_{\text{ratio}} \times \text{Launch MSRP}$$
- **Feature Space**:
  - *Numerical*: `vehicle_age`, `km_driven`, `km_per_year`, `engine_cc`, `max_power_bhp`, `power_to_displacement_ratio`.
  - *Categorical*: `brand`, `fuel_type`, `transmission`, `brand_tier`.
- **Model**: Gradient Boosted Decision Trees (`XGBRegressor`) with histogram-based binning.

### 2.2 Branch 2: Mechanical Health Sequence Model (Bi-LSTM)
- **Objective**: Ingest operational telemetry streams to detect latent mechanical degradation and Remaining Useful Life (RUL).
- **Data Source**: High-frequency operational telemetry (14 active, variance-informative sensor channels).
- **Sequence Formulation**: A sliding window tensor $X_{\text{seq}} \in \mathbb{R}^{B \times 30 \times 14}$ representing 30 consecutive operational cycles.
- **Target**: Piecewise linear degradation target clipped at 125 cycles normalized into $\text{MHI} \in [0.0, 1.0]$.
- **Architecture**: 2-layer Bi-Directional LSTM with hidden size $h = 64$ per direction ($2h = 128$), dropout, linear projection head, and sigmoid activation.

### 2.3 Branch 3: Multimodal Service Log Regressor (DistilBERT + MLP)
- **Objective**: Quantify unstructured technician commentary and physical inspection metrics into an objective Condition Score ($\text{CS} \in [0.0, 1.0]$).
- **Modalities**:
  - *Unstructured Text*: Diagnostic notes, work order remarks, and trouble codes.
  - *Tabular Checklist*: 4 physical metrics: `tire_tread_mm`, `brake_pad_wear_pct`, `fluid_condition`, `service_adherence_ratio`.
- **Architecture**:
  - Pre-trained `distilbert-base-uncased` extracting $[CLS]$ token representation $e_{\text{text}} \in \mathbb{R}^{768}$.
  - Tabular MLP projecting physical metrics into $e_{\text{tab}} \in \mathbb{R}^{32}$.
  - Fused Multi-Modal Embedding $e_{\text{fused}} = [ e_{\text{text}} \,\|\, e_{\text{tab}} ] \in \mathbb{R}^{800}$.
  - Multilayer regression head producing $\text{CS} \in [0.0, 1.0]$.

---

## 3. Late-Fusion Elasticity Formulation

The outputs of the three branches combine via a linear elasticity model centered on market neutrality:

$$M = 1.0 + w_1 \cdot (\text{MHI} - 0.5) + w_2 \cdot (\text{CS} - 0.5)$$
$$P_{\text{final}} = P_{\text{base}} \times M$$

Where:
- $w_1 = 0.15$ is the **Mechanical Telemetry Elasticity Weight**.
- $w_2 = 0.15$ is the **Physical Condition Elasticity Weight**.
- $0.5$ is the **Neutral Market Baseline** representing average wear.
- $\Delta_{\text{INR}} = P_{\text{final}} - P_{\text{base}}$ is the condition-driven valuation premium or penalty.
- $\Delta_{\%} = (M - 1.0) \times 100\%$.

### Defect Risk Assessment
$$\text{Defect Risk Score} = 1.0 - \left( 0.5 \cdot \text{MHI} + 0.5 \cdot \text{CS} \right)$$

- **Pristine**: $\text{MHI} \ge 0.70$ and $\text{CS} \ge 0.70 \implies \text{Defect Risk} < 0.30$.
- **Well-Maintained**: $0.50 \le \text{MHI}, \text{CS} < 0.70 \implies 0.30 \le \text{Defect Risk} < 0.50$.
- **Baseline**: Neutral market average condition ($\text{MHI} \approx 0.50, \text{CS} \approx 0.50, M = 1.000\text{x}$).
- **Minor Faults**: $0.30 \le \text{MHI} < 0.50$ or $0.30 \le \text{CS} < 0.50 \implies 0.50 \le \text{Defect Risk} < 0.70$.
- **Critical Defects**: $\text{MHI} < 0.30$ or $\text{CS} < 0.30 \implies \text{Defect Risk} \ge 0.70$.

---

## 4. Empirical Evaluation Across Vehicle Profiles

Evaluated on a representative mid-size Indian test vehicle ($P_{\text{base}} = \text{₹8,00,000}$):

| Vehicle Condition Profile | MHI (Branch 2) | CS (Branch 3) | Multiplier ($M$) | Baseline FMV | Adjusted Valuation | Net Adjustment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pristine** | 0.95 | 0.96 | **1.137x** | ₹8,00,000 | **₹9,09,200** | +₹1,09,200 (+13.7%) |
| **Well-Maintained** | 0.72 | 0.75 | **1.071x** | ₹8,00,000 | **₹8,56,400** | +₹56,400 (+7.1%) |
| **Baseline** | 0.50 | 0.50 | **1.000x** | ₹8,00,000 | **₹8,00,000** | ₹0 (0.0%) |
| **Minor Faults** | 0.38 | 0.40 | **0.967x** | ₹8,00,000 | **₹7,73,600** | -₹26,400 (-3.3%) |
| **Critical Defects** | 0.12 | 0.15 | **0.891x** | ₹8,00,000 | **₹7,12,400** | -₹87,600 (-11.0%) |

**Closing the Market Gap**: The 24.7% condition-driven valuation spread counteracts information asymmetry by rewarding preserved vehicles while applying fair discounts to degraded units.

---

## 5. Explainable AI (XAI) & Diagnostics

1. **Tabular Feature Attribution (Tree SHAP)**:
   - Quantifies individual feature impact (e.g., `vehicle_age`, `km_driven`, `power_to_displacement_ratio`) on baseline depreciation.
2. **Transformer Token Saliency (Input $\times$ Gradient)**:
   - Computes embedding gradients $\text{Saliency}(t) = \nabla_{e_t} \mathcal{L} \cdot e_t$ to highlight positive health markers (*regular*, *passed*, *nominal*) and risk markers (*blown*, *sludge*, *overheating*).
3. **Sensor Telemetry Trajectory Anomaly Radar**:
   - Compares 14-channel operational averages against healthy baselines to pinpoint anomalous subsystems.

---

## 6. Repository Layout

```
VahanaVeritas/
├── app.py                         # Streamlit Interactive Valuation Cockpit
├── models/
│   ├── market_regressor.py        # Branch 1 XGBoost Model Wrapper & Preprocessor
│   ├── mechanical_lstm.py         # Branch 2 Bi-LSTM PyTorch Model & Sequence Generator
│   ├── multimodal_nlp.py          # Branch 3 DistilBERT + MLP Multimodal Architecture
│   └── fusion_engine.py           # Late-Fusion Mathematics, Benchmarks & Risk Logic
├── utils/
│   ├── data_loader.py             # Dataset helpers, brand presets & MSRP tables
│   ├── explainability.py          # SHAP attribution & NLP token saliency visualizers
│   └── report_generator.py        # Inspection certificate & export utilities
├── scripts/
│   ├── train_and_export.py        # End-to-end model training script
│   └── generate_mock_telemetry.py # Telemetry sequence synthesizer
├── tests/
│   └── test_inference.py          # Late-Fusion mathematical verification test suite
├── requirements.txt               # Pinned Python dependencies
├── PROJECT_BRIEF.md               # Detailed technical project specification
└── README.md                      # Project documentation and guide
```

---

## 7. Installation & Quick Start

### Prerequisites
- Python 3.10, 3.11, or 3.12 (Python 3.13 supported)
- Virtual environment recommended (`venv` or `conda`)

### Step 1: Clone and Navigate
```bash
cd VahanaVeritas
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Inference Tests
Verify that mathematical formulations and boundary constraints hold:
```bash
python -m unittest tests/test_inference.py
```

### Step 4: Launch the Valuation Cockpit
```bash
streamlit run app.py
```

Open your browser at **`http://localhost:8501`** to interact with the multi-modal valuation cockpit.
