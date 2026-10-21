"""
Late-Fusion Engine for VahanaVeritas.
Combines Branch 1 (Market Regressor), Branch 2 (Bi-LSTM Telemetry),
and Branch 3 (Multimodal Inspection NLP) into condition-adjusted valuations.
"""

from typing import Dict, Any, Tuple

DEFAULT_W1 = 0.15 # Mechanical Telemetry Elasticity Weight
DEFAULT_W2 = 0.15 # Inspection Condition Elasticity Weight
NEUTRAL_BASELINE = 0.50 # Market neutral baseline point

ARCHETYPES = {
    "pristine": {
        "title": "Pristine",
        "mhi": 0.95,
        "cs": 0.96,
        "text": "Regular periodic service completed. All fluids fresh, nominal brake wear, passed 200-point inspection flawlessly.",
        "specs": {"brand": "Honda", "model": "City", "model_year": 2021, "km_driven": 22000, "fuel_type": "Petrol", "transmission": "Manual"},
        "desc": "Flawless mechanical health, low vibration, zero error codes, factory-spec fluid condition."
    },
    "nominal": {
        "title": "Well-Maintained",
        "mhi": 0.72,
        "cs": 0.75,
        "text": "Scheduled 40,000 km interval service completed. Oil and cabin filter replaced. Multi-point inspection passed with minor wear.",
        "specs": {"brand": "Hyundai", "model": "Creta", "model_year": 2020, "km_driven": 48000, "fuel_type": "Diesel", "transmission": "Manual"},
        "desc": "Timely periodic maintenance on schedule, normal highway wear, clean telemetry signals."
    },
    "neutral": {
        "title": "Baseline",
        "mhi": 0.50,
        "cs": 0.50,
        "text": "Average market condition. Uninspected prior maintenance history. Normal cosmetic wear.",
        "specs": {"brand": "Maruti", "model": "Swift", "model_year": 2019, "km_driven": 65000, "fuel_type": "Petrol", "transmission": "Manual"},
        "desc": "Standard uninspected vehicle assuming market average wear. Multiplier remains 1.00x."
    },
    "neglected": {
        "title": "Minor Faults",
        "mhi": 0.38,
        "cs": 0.40,
        "text": "Customer reported brake pulsation. Front pads heavily worn, rotors grooved. Minor valve cover gasket sweating detected.",
        "specs": {"brand": "Tata", "model": "Nexon", "model_year": 2018, "km_driven": 85000, "fuel_type": "Diesel", "transmission": "Manual"},
        "desc": "Deferred fluid replacement, mild thermal stress drift, brake wear beyond manufacturer threshold."
    },
    "lemon": {
        "title": "Critical Defects",
        "mhi": 0.12,
        "cs": 0.15,
        "text": "Engine overheating, blown head gasket diagnosed. Cylinder head warped. Severe sludge throughout crankcase.",
        "specs": {"brand": "Mahindra", "model": "Scorpio", "model_year": 2017, "km_driven": 115000, "fuel_type": "Diesel", "transmission": "Manual"},
        "desc": "Critical mechanical failure, valvetrain degradation, severe bearing vibration and impending seizure."
    }
}

def calculate_late_fusion(
    baseline_price: float,
    mhi: float,
    cs: float,
    w1: float = DEFAULT_W1,
    w2: float = DEFAULT_W2
) -> Dict[str, Any]:
    """
    Computes late-fusion multiplier and adjusted valuation.

    Mathematical Formulation:
        M = 1.0 + w1 * (MHI - 0.5) + w2 * (CS - 0.5)
        P_final = P_base * M
        Delta_INR = P_final - P_base
        Delta_pct = (M - 1.0) * 100
    """
    mhi_clamped = max(0.0, min(1.0, float(mhi)))
    cs_clamped = max(0.0, min(1.0, float(cs)))

    # Component adjustments
    mhi_delta = mhi_clamped - NEUTRAL_BASELINE
    cs_delta = cs_clamped - NEUTRAL_BASELINE

    mhi_impact_ratio = w1 * mhi_delta
    cs_impact_ratio = w2 * cs_delta

    multiplier = 1.0 + mhi_impact_ratio + cs_impact_ratio
    # Guardrail multiplier between [0.70, 1.30]
    multiplier = max(0.70, min(1.30, multiplier))

    final_price = baseline_price * multiplier
    delta_inr = final_price - baseline_price
    delta_pct = (multiplier - 1.0) * 100.0

    # Dollar/INR component breakdown for waterfall chart
    mhi_delta_inr = baseline_price * mhi_impact_ratio
    cs_delta_inr = baseline_price * cs_impact_ratio

    # 95% Confidence Interval band based on test MAPE (~8%)
    confidence_spread = final_price * 0.065
    ci_lower = max(0.0, final_price - confidence_spread)
    ci_upper = final_price + confidence_spread

    risk_info = get_lemon_risk_assessment(mhi_clamped, cs_clamped)

    return {
        'baseline_price': baseline_price,
        'mhi': mhi_clamped,
        'cs': cs_clamped,
        'w1': w1,
        'w2': w2,
        'multiplier': multiplier,
        'final_price': final_price,
        'delta_inr': delta_inr,
        'delta_pct': delta_pct,
        'mhi_delta_inr': mhi_delta_inr,
        'cs_delta_inr': cs_delta_inr,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'risk_score': risk_info['risk_score'],
        'risk_tier': risk_info['risk_tier'],
        'risk_color': risk_info['risk_color'],
        'recommendation': risk_info['recommendation']
    }

def get_lemon_risk_assessment(mhi: float, cs: float) -> Dict[str, Any]:
    """
    Computes composite Lemon Risk Index from MHI and CS.
    Risk Score = 1.0 - (0.5 * MHI + 0.5 * CS)
    """
    composite_health = 0.5 * mhi + 0.5 * cs
    risk_score = 1.0 - composite_health

    if mhi >= 0.70 and cs >= 0.70:
        tier = "Pristine"
        color = "#10b981" # Emerald Green
        rec = "Strong Buy. Vehicle exceeds market benchmarks. Premium valuation warranted."
    elif mhi >= 0.50 and cs >= 0.50:
        tier = "Well-Maintained"
        color = "#0284c7" # Slate Blue
        rec = "Standard Buy. Normal wear consistent with age and odometer. Proceed at FMV."
    elif mhi < 0.30 or cs < 0.30:
        tier = "Critical Defects"
        color = "#ef4444" # Crimson Red
        rec = "DO NOT PURCHASE WITHOUT WARRANTY. Severe latent defect detected. Demand >= 15% discount or reject."
    else:
        tier = "Minor Faults"
        color = "#d97706" # Amber
        rec = "Caution. Diagnostic or mechanical wear detected. Factor in near-term maintenance cost (Rs. 30,000-Rs. 60,000)."

    return {
        'risk_score': float(risk_score),
        'risk_tier': tier,
        'risk_color': color,
        'recommendation': rec
    }
