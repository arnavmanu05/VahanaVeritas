# VahanaVeritas: Multimodal Used Vehicle Valuation & Condition Assessment Engine

## 1. Executive Summary & Problem Formulation
In emerging automotive markets such as India, the pre-owned vehicle ecosystem is severely plagued by **information asymmetry**—the classical economic phenomenon formalized in George Akerlof's *The Market for Lemons*. Traditional valuation algorithms rely almost exclusively on macro tabular indicators: vehicle age, odometer reading (kilometer count), fuel type, and brand. 

However, two identical 2019 Hyundai Cretas with 60,000 km on the odometer can differ dramatically in underlying residual value:
- **Vehicle A**: Highway driven, strict OEM periodic service compliance, pristine mechanical compression, zero diagnostic error codes.
- **Vehicle B**: City-driven under thermal stress, delayed oil changes leading to valvetrain sludge, resurfaced brake pads, worn suspension bushings, or silent head-gasket failure.

Under conventional tabular-only pricing models, both cars receive identical valuations. Buyers face adverse selection and discount all offerings by an average **18% to 25% "lemon discount"**, punishing honest sellers and disincentivizing proactive maintenance.

**VahanaVeritas** solves this fundamental failure mode through a **Tri-Branch Multimodal Late-Fusion Architecture** that ingests:
1. **Macro Market Tabular Regressors** (Indian resale market dynamics).
2. **High-Frequency Sensor Telemetry Sequences** (Mechanical Health Index via Bi-Directional LSTM).
3. **Multimodal Diagnostic Records & Inspection Text** (Condition Score via DistilBERT + MLP Early Fusion).

---

## 2. Mathematical System Architecture

The overall valuation pipeline consists of three decoupled branch models unified by an **interpretable Late-Fusion Elasticity Engine**:

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
- **Objective**: Establish baseline Fair Market Value ($P_{base}$) reflecting macro depreciation patterns across Indian car markets.
- **Target Formulation**: To prevent price scale distortion across ultra-luxury vs economy hatchbacks, the model predicts the dimensionless **Depreciation Ratio** $y_{ratio} \in [0.10, 1.10]$:
  $$y_{ratio} = \frac{\text{Selling Price}}{\text{Launch MSRP}}$$
  $$P_{base} = \hat{y}_{ratio} \times \text{Launch MSRP}$$
- **Feature Space**:
  - Continuous ($x_{num}$): `vehicle_age`, `km_driven`, `km_per_year`, `engine_cc`, `max_power_bhp`, `power_to_displacement_ratio = max_power_bhp / (engine_cc + 1e-5)`
  - Categorical ($x_{cat}$): `brand`, `fuel_type`, `transmission`, `brand_tier` (Tier 1 High Retention, Tier 2 Mainstream, Tier 3 Luxury, Tier 4 Other).
- **Transformation**: Numerical standard scaling $\mathcal{N}(0, 1)$ + Categorical One-Hot Encoding.
- **Model**: Regularized Gradient Boosted Decision Trees (`XGBRegressor`) with histogram-based binning (`tree_method='hist'`).

### 2.2 Branch 2: Mechanical Health Sequence Model (Bi-LSTM)
- **Objective**: Ingest multi-channel sensor telemetry streams to detect latent mechanical degradation, wear-and-tear, and remaining useful life (RUL).
- **Data Source**: High-frequency operational telemetry (NASA C-MAPSS FD001 engine/vehicle degradation data), using 14 active, variance-informative sensor channels after dropping 7 invariant channels.
- **Preprocessing**: Channel-wise min-max normalization into $[0.0, 1.0]$.
- **Sequence Formulation**: A sliding window tensor $X_{\text{seq}} \in \mathbb{R}^{B \times 30 \times 14}$ representing 30 consecutive operational cycles.
- **Target Formulation**: Piecewise linear degradation target clipped at 125 cycles:
  $$\text{RUL}_{\text{clipped}} = \min(\text{RUL}, 125)$$
  $$\text{MHI} = \frac{\text{RUL}_{\text{clipped}}}{125.0} \in [0.0, 1.0]$$
- **Neural Architecture**:
  - Two-layer Bi-Directional LSTM: Hidden dimension $h = 64$ per direction ($2h = 128$), dropout $p = 0.20$.
  - Sequence representation extracted from the final bidirectional hidden state: $h_T = [ \overrightarrow{h_T} \,\|\, \overleftarrow{h_1} ] \in \mathbb{R}^{128}$.
  - Dense Projection Head: $\text{Linear}(128 \to 32) \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{Linear}(32 \to 1) \to \text{Sigmoid}$.
  - Sigmoid ensures strict boundary $\text{MHI} \in (0.0, 1.0)$.

### 2.3 Branch 3: Multimodal Service Log NLP Regressor (DistilBERT + MLP)
- **Objective**: Quantify qualitative inspector notes and physical inspection wear metrics into an objective Condition Score ($\text{CS} \in [0.0, 1.0]$).
- **Dual Modalities**:
  - *Unstructured Text*: Technician diagnostic notes, work orders, fault code descriptions, service log remarks.
  - *Quantitative Tabular Checklist*: 4 physical metrics: `tire_tread_mm` (1.0 - 8.0 mm), `brake_pad_wear_pct` (10% - 100%), `fluid_condition` (Good: 1.0, Fair: 0.65, Contaminated: 0.20), `service_adherence_ratio` (0.10 - 1.00).
- **Early-Fusion Hybrid Architecture**:
  - Transformer text encoder: Pre-trained `distilbert-base-uncased` extracting $[CLS]$ token representation $e_{\text{text}} \in \mathbb{R}^{768}$. (First 4 transformer layers frozen to ensure real-time latency and prevent catastrophic forgetting).
  - Tabular MLP: $\text{Linear}(4 \to 32) \to \text{ReLU} \to \text{Dropout}(0.2)$ producing $e_{\text{tab}} \in \mathbb{R}^{32}$.
  - Fused Multi-Modal Embedding: $e_{\text{fused}} = [ e_{\text{text}} \,\|\, e_{\text{tab}} ] \in \mathbb{R}^{800}$.
  - Fusion Head: $\text{Linear}(800 \to 128) \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{Linear}(128 \to 32) \to \text{ReLU} \to \text{Linear}(32 \to 1) \to \text{Sigmoid}$.
  - Output: $\text{CS} \in [0.0, 1.0]$.

### 2.4 Late-Fusion Elasticity Engine
The outputs of the three branches are combined via an elasticity formulation:

$$M = 1.0 + w_1 \cdot (\text{MHI} - 0.5) + w_2 \cdot (\text{CS} - 0.5)$$
$$P_{\text{final}} = P_{\text{base}} \times M$$

Where:
- $w_1 = 0.15$ is the **Mechanical Telemetry Elasticity Weight**.
- $w_2 = 0.15$ is the **Physical Condition Elasticity Weight**.
- $0.5$ is the **Neutral Market Baseline** representing an uninspected vehicle with average wear.
- $\Delta_{\text{INR}} = P_{\text{final}} - P_{\text{base}}$ is the condition-driven valuation premium or penalty.
- $\Delta_{\%} = (M - 1.0) \times 100\%$.

#### Composite Defect Risk Index
$$\text{Defect Risk Score} = 1.0 - \left( 0.5 \cdot \text{MHI} + 0.5 \cdot \text{CS} \right)$$
- **Pristine**: $\text{MHI} \ge 0.70$ and $\text{CS} \ge 0.70 \implies \text{Risk} < 0.30$.
- **Well-Maintained**: $0.50 \le \text{MHI}, \text{CS} < 0.70 \implies 0.30 \le \text{Risk} < 0.50$.
- **Minor Faults**: $0.30 \le \text{MHI} < 0.50$ or $0.30 \le \text{CS} < 0.50 \implies 0.50 \le \text{Risk} < 0.70$.
- **Critical Defects**: $\text{MHI} < 0.30$ or $\text{CS} < 0.30 \implies \text{Risk} \ge 0.70$.

---

## 3. Empirical Benchmarks & Performance Metrics

Empirical evaluations conducted across independent held-out test splits (80/20 train/test split with deterministic seeding):

### Table 1: Multi-Branch Empirical Performance Benchmark
| Branch / Sub-System | Modality / Data Source | Target Output | $R^2$ Score | Primary Test Error |
| :--- | :--- | :--- | :--- | :--- |
| **Branch 1: Tabular Market Regressor** | Macro Resale Listings (CarDekho / Cars24) | Depreciation Ratio ($y_{ratio}$) | **0.9203** | MAE: **₹99,367** (MAPE: **17.99%**, RMSE: ₹168,333) |
| **Branch 2: Mechanical Health Sequence Model** | 14-Channel Telemetry (NASA C-MAPSS FD001) | Mechanical Health Index ($\text{MHI} \in [0, 1]$) | **0.8987** | RMSE: **0.1065** (MAE: **0.0824**) |
| **Branch 3: Multimodal Service Log Regressor** | Inspection Checklist + Diagnostic Text | Condition Score ($\text{CS} \in [0, 1]$) | **0.8953** | RMSE: **0.0908** (MAE: **0.0708**) |

### Table 2: Late-Fusion Elasticity Resolution Across Archetypes
Demonstration on a representative mid-size Indian test vehicle (e.g., Hyundai Creta / Honda City, $P_{\text{base}} = \text{₹8,00,000}$):

| Vehicle Health Archetype | MHI (Branch 2) | CS (Branch 3) | Late-Fusion Multiplier | Unadjusted Baseline | Final Condition-Adjusted Value | Valuation Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pristine** | 0.95 | 0.96 | **1.137x** | ₹8,00,000 | **₹9,09,200** | +₹1,09,200 (+13.7%) |
| **Well-Maintained** | 0.72 | 0.75 | **1.071x** | ₹8,00,000 | **₹8,56,400** | +₹56,400 (+7.1%) |
| **Baseline** | 0.50 | 0.50 | **1.000x** | ₹8,00,000 | **₹8,00,000** | ₹0 (0.0%) |
| **Minor Faults** | 0.38 | 0.40 | **0.967x** | ₹8,00,000 | **₹7,73,600** | -₹26,400 (-3.3%) |
| **Critical Defects** | 0.12 | 0.15 | **0.891x** | ₹8,00,000 | **₹7,12,400** | -₹87,600 (-11.0%) |

**Closing the Lemon Gap**: The dynamic spread from **-11.0%** to **+13.7%** spans **24.7%**, successfully counteracting the uninformative market spread.

---

## 4. Explainable AI (XAI) & Diagnostics Framework

1. **Tabular Feature Attribution (Tree SHAP)**:
   - Quantifies the directional contribution of each feature to the depreciation ratio.
   - Primary drivers: `vehicle_age`, `km_driven`, `brand_tier`, and `power_to_displacement_ratio`.
2. **NLP Token Saliency (Input $\times$ Gradient Attribution)**:
   - Computes gradients with respect to DistilBERT word embedding vectors:
     $$\text{Saliency}(t) = \nabla_{e_t} \mathcal{L} \cdot e_t$$
   - Highlights keywords indicating mechanical health (e.g., *"regular"*, *"passed"*, *"nominal"*) in green, and critical defect tokens (e.g., *"blown"*, *"sludge"*, *"overheating"*, *"cracked"*) in red.
3. **Sensor Telemetry Trajectory Anomaly Radar**:
   - Visualizes multi-channel operational deviation from healthy baselines over the 30-cycle observation window.
4. **Segmented Residual Diagnostics**:
   - Age cohort error stratification: Near-new vehicles (0-4 yrs: MAPE 12.37%), mid-life (5-8 yrs: MAPE 15.74%), mature (>8 yrs: MAPE 21.1%).
   - Demonstrates heteroscedasticity management across brand tiers and mileage bands.

---

## 5. Repository Structure & Execution Instructions

```
c:\Users\arnav\VahanaVeritas\
├── app.py                         # Streamlit UI & Interactive Valuation Cockpit
├── models/
│   ├── __init__.py
│   ├── market_regressor.py        # Branch 1 XGBoost Model Wrapper & Preprocessor
│   ├── mechanical_lstm.py         # Branch 2 Bi-LSTM PyTorch Definition & Inference
│   ├── multimodal_nlp.py          # Branch 3 DistilBERT + MLP Multimodal Architecture
│   └── fusion_engine.py           # Late-Fusion Math, Archetypes & Risk Diagnostics
├── utils/
│   ├── __init__.py
│   ├── data_loader.py             # Dataset helpers, brand presets & MSRP tables
│   ├── explainability.py          # SHAP attribution & NLP token saliency visualizers
│   └── report_generator.py        # PDF/Text Inspection & Valuation Report Exporter
├── scripts/
│   ├── train_and_export.py        # End-to-end model training & artifact generation
│   └── generate_mock_telemetry.py # Utility to synthesize test sequence tensors
├── tests/
│   └── test_inference.py          # End-to-end inference & Late-Fusion verification suite
├── requirements.txt               # Pinned Python package dependencies
└── PROJECT_BRIEF.md               # Complete System Architectural Specification
```

### Local Execution Commands
```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Run end-to-end verification tests
python -m unittest tests/test_inference.py

# 3. Launch the Streamlit application
streamlit run app.py
```
