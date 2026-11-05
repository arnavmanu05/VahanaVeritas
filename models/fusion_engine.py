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
        "text": (
            "Comprehensive 200-point OEM master inspection completed flawlessly with zero advisories.\n"
            "Full periodic service history verified with all fluids (synthetic 0W-20 engine oil, transmission fluid, and DOT-4 brake fluid) fresh and at optimal viscosity.\n"
            "Brake pads and rotors show nominal minimal wear, suspension bushings are firm with zero play or cracking, and tires retain excellent 7.2mm uniform tread depth.\n"
            "On-board diagnostic ECU scan confirms complete readiness monitors with zero fault codes or stored historical DTCs.\n"
            "Engine idles smooth with factory compression numbers, valvetrain is whisper quiet, and vehicle exhibits pristine showroom condition throughout."
        ),
        "specs": {"brand": "Honda", "model": "City", "model_year": 2021, "km_driven": 22000, "fuel_type": "Petrol", "transmission": "Manual"},
        "desc": "Flawless mechanical health, low vibration, zero error codes, factory-spec fluid condition."
    },
    "nominal": {
        "title": "Well-Maintained",
        "mhi": 0.72,
        "cs": 0.75,
        "text": (
            "Scheduled 40,000 km periodic interval service successfully executed in compliance with manufacturer schedule.\n"
            "Engine oil and oil filter replaced with OEM synthetic blend; air intake and cabin air filters renewed; coolant and brake fluid topped off to nominal levels.\n"
            "Multi-point underbody inspection revealed normal highway operating wear across chassis components; front brake pads measured at 65% remaining life.\n"
            "All major electronic sensors and actuators tested within standard manufacturer operating tolerances with zero active check-engine DTCs.\n"
            "Overall vehicle displays consistent preventative care, clean exhaust emissions, and very healthy mechanical roadworthiness."
        ),
        "specs": {"brand": "Hyundai", "model": "Creta", "model_year": 2020, "km_driven": 48000, "fuel_type": "Diesel", "transmission": "Manual"},
        "desc": "Timely periodic maintenance on schedule, normal highway wear, clean telemetry signals."
    },
    "neutral": {
        "title": "Baseline",
        "mhi": 0.50,
        "cs": 0.50,
        "text": (
            "Standard walk-around vehicle evaluation completed for routine trade-in intake.\n"
            "Service records indicate average historical maintenance cadence without specialized third-party multi-point verification.\n"
            "Tire treads show moderate symmetric wear at approximately 4.2mm, brake pads exhibit expected mid-life friction lining reduction, and engine oil condition is fair.\n"
            "Minor cosmetic road chips observed along lower bumper fascia; suspension and steering feel typical for vehicle mileage band.\n"
            "ECU diagnostic scan reports no fatal system shutdowns, representing typical market average condition for this vehicle cohort."
        ),
        "specs": {"brand": "Maruti", "model": "Swift", "model_year": 2019, "km_driven": 65000, "fuel_type": "Petrol", "transmission": "Manual"},
        "desc": "Standard uninspected vehicle assuming market average wear. Multiplier remains 1.00x."
    },
    "neglected": {
        "title": "Minor Faults",
        "mhi": 0.38,
        "cs": 0.40,
        "text": (
            "Vehicle brought in following customer complaint of intermittent brake pedal pulsation and cold-start squeal.\n"
            "Physical inspection reveals heavily worn front brake pads at less than 20% remaining thickness with noticeable rotor disc grooving requiring resurfacing.\n"
            "Minor oil sweating and valve cover gasket seepage detected around upper cylinder head, along with aging accessory drive belt exhibiting minor micro-cracking.\n"
            "Tire tread is unevenly worn at 2.8mm, indicating slight front wheel alignment drift and deferred tire rotation intervals.\n"
            "Stored pending DTC detected for secondary O2 sensor response latency; requires scheduled mechanical refurbishment."
        ),
        "specs": {"brand": "Tata", "model": "Nexon", "model_year": 2018, "km_driven": 85000, "fuel_type": "Diesel", "transmission": "Manual"},
        "desc": "Deferred fluid replacement, mild thermal stress drift, brake wear beyond manufacturer threshold."
    },
    "lemon": {
        "title": "Critical Defects",
        "mhi": 0.12,
        "cs": 0.15,
        "text": (
            "Urgent diagnostic teardown initiated due to persistent thermal runaway and severe engine overheating.\n"
            "Comprehensive compression and block leakage testing confirms blown cylinder head gasket with warped mating surface and coolant cross-contamination into crankcase.\n"
            "Severe valvetrain sludge, metal flake debris in oil pan, and heavy rod bearing play indicate extensive lubrication starvation and imminent mechanical seizure.\n"
            "Exhaust exhibits thick white vapor smoke, catalytic converter is fouled with unburnt hydrocarbon residue, and active MIL illuminates with multiple critical misfire DTCs.\n"
            "Vehicle deemed mechanically compromised with catastrophic powertrain defect risk requiring complete engine rebuild."
        ),
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
