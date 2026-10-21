import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any

def render_token_saliency_html(tokens: List[str], scores: List[float]) -> str:
    """
    Renders styled HTML tokens with green/red attribution highlights for executive light theme.
    Positive scores = healthy indicators (emerald)
    Negative scores = defects / degradation (crimson)
    """
    if not tokens:
        return "<em style='color: #64748b;'>No token attributions available.</em>"

    max_score = max(max([abs(s) for s in scores]), 1e-4)

    html_parts = []
    for tok, score in zip(tokens, scores):
        norm_score = score / max_score
        intensity = min(1.0, abs(norm_score))
        alpha = 0.15 + (intensity * 0.45)

        if score > 0.05:
            # Positive indicator -> Emerald green on light background
            bg = f"rgba(16, 185, 129, {alpha:.2f})"
            color = "#065f46"
            border = "rgba(16, 185, 129, 0.6)"
            title = f"+{score:.3f} (Health indicator)"
        elif score < -0.05:
            # Negative defect indicator -> Crimson red on light background
            bg = f"rgba(239, 68, 68, {alpha:.2f})"
            color = "#991b1b"
            border = "rgba(239, 68, 68, 0.6)"
            title = f"{score:.3f} (Defect risk)"
        else:
            # Neutral token
            bg = "#f1f5f9"
            color = "#475569"
            border = "#e2e8f0"
            title = f"{score:.3f} (Neutral)"

        clean_tok = tok.replace("##", "").replace("<", "&lt;").replace(">", "&gt;")
        span = (
            f"<span title='{title}' style='"
            f"display: inline-block; margin: 3px 2px; padding: 4px 9px; "
            f"border-radius: 4px; background: {bg}; color: {color}; "
            f"border: 1px solid {border}; font-size: 0.88rem; font-family: 'JetBrains Mono', monospace; "
            f"font-weight: 600; letter-spacing: -0.2px; cursor: default;'>"
            f"{clean_tok}</span>"
        )
        html_parts.append(span)

    container = (
        "<div style='line-height: 2.2; padding: 18px; background: #ffffff; "
        "border: 1px solid #e2e8f0; border-radius: 8px; box-shadow: 0 2px 8px -1px rgba(15, 23, 42, 0.04);'>"
        + "".join(html_parts) +
        "</div>"
    )
    return container

def build_waterfall_figure(fusion_result: Dict[str, Any]):
    """
    Constructs an interactive Plotly Waterfall valuation decomposition chart with light executive palette.
    """
    import plotly.graph_objects as go

    base = fusion_result['baseline_price']
    mhi_inr = fusion_result['mhi_delta_inr']
    cs_inr = fusion_result['cs_delta_inr']
    final = fusion_result['final_price']

    fig = go.Figure(go.Waterfall(
        name="Valuation Dynamics",
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["1. Market Baseline FMV", "2. Mechanical Telemetry", "3. Physical Inspection", "4. Condition-Adjusted FMV"],
        textposition="outside",
        text=[f"Rs. {base:,.0f}", f"{'+' if mhi_inr >= 0 else ''}Rs. {mhi_inr:,.0f}", f"{'+' if cs_inr >= 0 else ''}Rs. {cs_inr:,.0f}", f"Rs. {final:,.0f}"],
        textfont=dict(family="JetBrains Mono, monospace", size=11, color="#0f172a"),
        y=[base, mhi_inr, cs_inr, final],
        connector={"line": {"color": "#94a3b8", "width": 1.5}},
        increasing={"marker": {"color": "#059669", "line": {"color": "#047857", "width": 1}}},
        decreasing={"marker": {"color": "#e11d48", "line": {"color": "#be123c", "width": 1}}},
        totals={"marker": {"color": "#0284c7", "line": {"color": "#0369a1", "width": 1}}}
    ))

    fig.update_layout(
        title={
            'text': "<b>Valuation Component Waterfall Breakdown</b> (INR)",
            'y': 0.95, 'x': 0.05, 'xanchor': 'left', 'yanchor': 'top',
            'font': {'family': 'Inter, sans-serif', 'size': 13, 'color': '#0f172a'}
        },
        template="plotly_white",
        paper_bgcolor="rgba(255, 255, 255, 0)",
        plot_bgcolor="rgba(255, 255, 255, 0)",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(
            tickfont=dict(family="Inter, sans-serif", size=11, color="#334155"),
            gridcolor="#f1f5f9"
        ),
        yaxis=dict(
            title=dict(text="Valuation (INR)", font=dict(family="Inter, sans-serif", size=12, color="#475569")),
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#64748b"),
            gridcolor="#f1f5f9"
        ),
        height=360
    )
    return fig

def build_telemetry_radar_figure(channel_averages: Dict[str, float]):
    """
    Generates an enterprise polar radar visualization of the 14 telemetry operational channels on light canvas.
    """
    import plotly.graph_objects as go

    # Friendly concise channel labels for crisp radar rendering
    short_labels = {
        'LPC Outlet Temp (S2)': 'S2: LPC Temp',
        'HPC Outlet Temp (S3)': 'S3: HPC Temp',
        'LPT Outlet Temp (S4)': 'S4: LPT Temp',
        'HPC Outlet Pressure (S7)': 'S7: HPC Press',
        'Physical Fan Speed (S8)': 'S8: Fan Speed',
        'Corrected Fan Speed (S9)': 'S9: Corr Fan',
        'HPC Static Pressure (S11)': 'S11: Static P',
        'Fuel Flow Ratio (S12)': 'S12: Fuel Ratio',
        'Corrected Core Speed (S13)': 'S13: Core Speed',
        'Bypass Duct Pressure (S14)': 'S14: Bypass P',
        'Bleed Enthalpy (S15)': 'S15: Bleed H',
        'High-Pressure Bleed (S17)': 'S17: HP Bleed',
        'HPT Coolant Bleed (S20)': 'S20: HPT Cool',
        'LPT Coolant Bleed (S21)': 'S21: LPT Cool'
    }

    categories = [short_labels.get(k, k) for k in channel_averages.keys()]
    values = list(channel_averages.values())

    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    healthy_baseline = [0.25] * len(values_closed)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=healthy_baseline,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(16, 185, 129, 0.10)',
        line=dict(color='#059669', width=1.5, dash='dash'),
        name='Nominal Baseline'
    ))

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(2, 132, 199, 0.15)',
        line=dict(color='#0284c7', width=2),
        name='Inspected Telemetry'
    ))

    fig.update_layout(
        title={
            'text': "<b>Operational Telemetry Deviation Radar</b>",
            'y': 0.96, 'x': 0.05, 'xanchor': 'left', 'yanchor': 'top',
            'font': {'family': 'Inter, sans-serif', 'size': 13, 'color': '#0f172a'}
        },
        polar=dict(
            bgcolor="rgba(248, 250, 252, 0.6)",
            radialaxis=dict(
                visible=True,
                range=[0, 1.0],
                gridcolor="#e2e8f0",
                tickfont=dict(family="JetBrains Mono, monospace", size=8, color="#64748b")
            ),
            angularaxis=dict(
                gridcolor="#e2e8f0",
                tickfont=dict(family="JetBrains Mono, monospace", size=8.5, color="#334155")
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.20,
            xanchor="center",
            x=0.5,
            font=dict(family="Inter, sans-serif", size=10, color="#334155"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#e2e8f0"
        ),
        template="plotly_white",
        paper_bgcolor="rgba(255, 255, 255, 0)",
        margin=dict(l=50, r=50, t=45, b=45),
        height=360
    )
    return fig

def compute_tree_shap_breakdown(market_pipeline, specs: Dict[str, Any], launch_msrp: float) -> pd.DataFrame:
    """
    Computes mathematically genuine Tree SHAP attributions directly from the trained XGBoost model
    using XGBoost's native Lundberg Tree SHAP algorithm (pred_contribs=True).
    Converts dimensionless depreciation ratio attributions into directional INR impacts.
    """
    import xgboost as xgb
    from models.market_regressor import engineer_features, NUM_COLS, CAT_COLS

    if market_pipeline is None:
        return pd.DataFrame()

    df_input = engineer_features(specs)
    preprocessor = market_pipeline.named_steps.get('preprocessor')
    regressor = market_pipeline.named_steps.get('regressor')

    if preprocessor is None or regressor is None:
        return pd.DataFrame()

    X_transformed = preprocessor.transform(df_input)
    dmat = xgb.DMatrix(X_transformed)
    contribs = regressor.get_booster().predict(dmat, pred_contribs=True)[0]

    cat_encoder = preprocessor.named_transformers_['cat']
    cat_names = list(cat_encoder.get_feature_names_out(CAT_COLS))
    all_names = NUM_COLS + cat_names
    feature_contribs = contribs[:-1]
    bias = contribs[-1]

    grouped = {
        'Vehicle Age': 0.0,
        'Kilometers Driven': 0.0,
        'Powertrain (Disp & Power)': 0.0,
        'Brand Tier Retention': 0.0,
        'Fuel Type': 0.0,
        'Transmission': 0.0
    }

    for name, val in zip(all_names, feature_contribs):
        if name == 'vehicle_age':
            grouped['Vehicle Age'] += val
        elif name in ['km_driven', 'km_per_year']:
            grouped['Kilometers Driven'] += val
        elif name in ['engine_cc', 'max_power_bhp', 'power_to_displacement_ratio']:
            grouped['Powertrain (Disp & Power)'] += val
        elif 'brand' in name:
            grouped['Brand Tier Retention'] += val
        elif 'fuel_type' in name:
            grouped['Fuel Type'] += val
        elif 'transmission' in name:
            grouped['Transmission'] += val

    rows = []
    for k, v in grouped.items():
        inr_val = float(v * launch_msrp)
        rows.append({
            'Feature Driver': k,
            'Attribution (INR Impact)': inr_val,
            'SHAP Ratio': float(v)
        })

    df_shap = pd.DataFrame(rows).sort_values(by='Attribution (INR Impact)', ascending=True)
    return df_shap

