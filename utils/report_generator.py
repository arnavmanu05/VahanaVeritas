import datetime
from typing import Dict, Any

def generate_inspection_certificate(
    specs: Dict[str, Any],
    fusion_result: Dict[str, Any],
    nlp_result: Dict[str, Any],
    mhi_result: Dict[str, Any]
) -> str:
    """
    Generates a formal, printable Markdown certificate summarizing the complete
    multi-branch valuation, condition scores, and lemon risk assessment.
    """
    now_str = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
    brand = specs.get('brand', 'Unknown')
    model = specs.get('model', 'Generic')
    year = specs.get('model_year', 2020)
    km = specs.get('km_driven', 50000)
    fuel = specs.get('fuel_type', 'Petrol')
    trans = specs.get('transmission', 'Manual')

    base_fmv = fusion_result['baseline_price']
    final_fmv = fusion_result['final_price']
    delta_inr = fusion_result['delta_inr']
    delta_pct = fusion_result['delta_pct']
    multiplier = fusion_result['multiplier']
    mhi = fusion_result['mhi']
    cs = fusion_result['cs']
    risk_tier = fusion_result['risk_tier']
    rec = fusion_result['recommendation']

    cert_text = f"""# VAHANAVERITAS AUTOMOTIVE VALUATION CERTIFICATE
**Certificate ID**: VV-{brand.upper()[:3]}-{year}-{int(km)}-{int(final_fmv % 10000):04d}  
**Timestamp**: {now_str}  
**Platform**: VahanaVeritas Multi-Modal Tri-Branch Fusion Engine  

---

## 1. VEHICLE IDENTITY & BASELINE SPECIFICATIONS
- **Make & Model**: {brand} {model} ({year})
- **Odometer Reading**: {km:,.0f} km
- **Powertrain**: {fuel} | {trans}
- **Engine Capacity**: {specs.get('engine_cc', 1497)} cc | Power: {specs.get('max_power_bhp', 100)} bhp
- **Original Launch MSRP**: ₹{fusion_result.get('launch_msrp', 1000000):,.0f}
- **Branch 1 Tabular Baseline FMV**: **₹{base_fmv:,.0f}**

---

## 2. MULTI-MODAL EMPIRICAL CONDITION METRICS

### A. Mechanical Telemetry Stream (Branch 2: Bi-LSTM)
- **Mechanical Health Index (MHI)**: **{mhi:.3f} / 1.000**
- **Telemetry Operational Status**: {mhi_result.get('status', 'Nominal')}
- **Sensor Deviation vs Baseline**: Evaluated across 14 dynamic operational channels (vibration, thermal enthalpy, exhaust pressure).

### B. Diagnostic & Physical Inspection (Branch 3: DistilBERT + MLP)
- **Physical Condition Score (CS)**: **{cs:.3f} / 1.000**
- **Inspection Grade**: {nlp_result.get('grade', 'Standard')}
- **Diagnostic Notes Evaluated**: *"{nlp_result.get('raw_text', 'No notes provided.')}"*

---

## 3. LATE-FUSION CONDITION ADJUSTMENT & FAIR MARKET PRICING

$$\\text{{Final Price}} = \\text{{Baseline FMV}} \\times [1.0 + w_1(\\text{{MHI}} - 0.5) + w_2(\\text{{CS}} - 0.5)]$$

- **Applied Late-Fusion Multiplier**: **{multiplier:.4f}x**
- **Baseline Market Fair Value**: ₹{base_fmv:,.0f}
- **Net Condition Adjustment**: {('+' if delta_inr >= 0 else '')}₹{delta_inr:,.0f} ({delta_pct:+.2f}%)
- **FINAL CERTIFIED CONDITION-ADJUSTED VALUE**: 
  ### **₹{final_fmv:,.0f}** (₹{final_fmv / 100000.0:.2f} Lakhs)
- **95% Confidence Valuation Band**: ₹{fusion_result.get('ci_lower', final_fmv * 0.95):,.0f} — ₹{fusion_result.get('ci_upper', final_fmv * 1.05):,.0f}

---

## 4. LEMON RISK AUDIT & TRANSACTION RECOMMENDATION
- **Lemon Risk Classification**: **{risk_tier}**
- **Certified Recommendation**:  
  > **{rec}**

*Disclaimer: This certificate is generated algorithmically utilizing calibrated machine learning pipelines (XGBoost, Bi-LSTM, and DistilBERT). Physical validation of safety-critical systems remains recommended prior to binding transaction agreements.*
"""
    return cert_text
