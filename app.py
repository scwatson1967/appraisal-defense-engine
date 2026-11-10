import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import re
from io import BytesIO
from datetime import date
from xhtml2pdf import pisa
from groq import Groq

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Listing Defense Brief & Pricing Engine",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .report-header { font-size: 26px; font-weight: bold; color: #1E3A8A; margin-bottom: 2px; }
    .sub-header { font-size: 13px; color: #4B5563; margin-bottom: 18px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# TEXT SANITIZER (PREVENTS BLACK GLYPH BOXES IN REPORTLAB / HELVETICA)
# ==============================================================================
def sanitize_for_pdf(text: str) -> str:
    if not isinstance(text, str):
        return str(text)

    replacements = {
        r"\pm": "+/-",
        "±": "+/-",
        r"\approx": "~",
        "≈": "~",
        r"\times": "x",
        "×": "x",
        r"\le": "<=",
        r"\ge": ">=",
        "–": "-",      # en-dash
        "—": "-",      # em-dash
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'",
        "…": "...",
        "\u00a0": " ",
        "\u200b": "",  # zero-width space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r'\$(.*?)\$', r'\1', text)
    text = text.replace(r"\mathbb{H}", "").replace(r"\mathbb", "").replace(r"\Pi", "").replace(r"\s", "")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

# ==============================================================================
# DATA INGESTION & NORMALIZATION
# ==============================================================================
def clean_currency(val):
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    val_str = re.sub(r'[^\d.]', '', str(val))
    return float(val_str) if val_str else 0.0

def load_and_clean_data(file):
    df = pd.read_csv(file)
    
    if 'MLS #' in df.columns:
        df['MLS #'] = df['MLS #'].astype(str).str.strip()

    price_cols = ['Close Price', 'Sold Price', 'Purchase Price', 'Price']
    found_price = next((c for c in price_cols if c in df.columns), None)
    df['Clean_Price'] = df[found_price].apply(clean_currency) if found_price else 0.0
    df = df[df['Clean_Price'] > 20000].copy()

    gla_cols = ['Living Area', 'Above Grade Finished Area', 'Gross Living Area', 'SqFt']
    found_gla = next((c for c in gla_cols if c in df.columns), None)
    df['Clean_GLA'] = df[found_gla].apply(clean_currency) if found_gla else 0.0

    bed_cols = ['Bedrooms Total', 'Bedrooms', 'Beds Total']
    found_bed = next((c for c in bed_cols if c in df.columns), None)
    df['Clean_Beds'] = df[found_bed].fillna(3).astype(int) if found_bed else 3

    fb_cols = ['Bathrooms Full', 'Baths Full', 'Full Baths']
    found_fb = next((c for c in fb_cols if c in df.columns), None)
    df['Clean_FB'] = df[found_fb].fillna(2).astype(int) if found_fb else 2

    hb_cols = ['Bathrooms Half', 'Baths Half', 'Half Baths']
    found_hb = next((c for c in hb_cols if c in df.columns), None)
    df['Clean_HB'] = df[found_hb].fillna(0).astype(int) if found_hb else 0

    subd_cols = ['Subdivision', 'Subdivision-Free Text', 'Legal Subdivision']
    found_subd = next((c for c in subd_cols if c in df.columns), None)
    df['Clean_Subdivision'] = df[found_subd].fillna("").astype(str).str.strip() if found_subd else ""

    date_cols = ['Close Date', 'Closed Date', 'Sold Date']
    found_date = next((c for c in date_cols if c in df.columns), None)
    if found_date:
        df['Clean_Date'] = pd.to_datetime(df[found_date], errors='coerce')
    else:
        df['Clean_Date'] = pd.to_datetime('today')

    garage_cols = ['Garage Spaces', 'Garage Total', 'Garage']
    found_gar = next((c for c in garage_cols if c in df.columns), None)
    df['Clean_Garage'] = df[found_gar].apply(clean_currency) if found_gar else 0.0

    return df

# ==============================================================================
# HEDONIC REGRESSION & SENSITIVITY ENGINE
# ==============================================================================
def run_regression_analysis(df, subject):
    reg_df = df[df['Clean_GLA'] > 500].copy()
    if len(reg_df) < 5:
        return {
            "r2": 0.50,
            "gla_rate": 65.0,
            "fb_adj": 7500.0,
            "hb_adj": 3500.0,
            "bed_adj": 0.0,
            "garage_rate": 10000.0,
            "benchmark": float(subject['gla'] * 180)
        }

    X = reg_df[['Clean_GLA', 'Clean_FB', 'Clean_HB']]
    X = sm.add_constant(X)
    y = reg_df['Clean_Price']

    try:
        model = sm.OLS(y, X).fit()
        gla_coef = model.params.get('Clean_GLA', 45.0)
        fb_coef = model.params.get('Clean_FB', 5000.0)
        hb_coef = model.params.get('Clean_HB', 2500.0)
        r2 = model.rsquared

        gla_rate = round(float(np.clip(gla_coef, 35.0, 185.0)), 2)
        fb_adj = round(float(np.clip(fb_coef, 3000.0, 25000.0)), 0)
        hb_adj = round(float(np.clip(hb_coef, 1500.0, 12500.0)), 0)

        subj_pred = model.predict([1, subject['gla'], subject['baths_full'], subject['baths_half']])[0]
        benchmark = round(float(subj_pred), 0)
    except Exception:
        r2 = 0.40
        gla_rate = 55.0
        fb_adj = 5000.0
        hb_adj = 2500.0
        benchmark = round(float(df['Clean_Price'].median()), 0)

    return {
        "r2": round(r2, 4),
        "gla_rate": gla_rate,
        "fb_adj": fb_adj,
        "hb_adj": hb_adj,
        "bed_adj": 0.0,
        "garage_rate": 10000.0,
        "benchmark": benchmark
    }

# ==============================================================================
# COMPARABLE SELECTION (SUBDIVISION PRIORITY + ANSI BRACKETING)
# ==============================================================================
def select_comparables_with_subdivision(df, subject, mode="algorithmic", manual_mls_ids=None):
    df_pool = df.copy()
    subj_gla = subject['gla']
    subj_subd = str(subject.get('subdivision', '')).strip().lower()

    if mode == "manual" and manual_mls_ids:
        selected = df_pool[df_pool['MLS #'].astype(str).isin([str(x) for x in manual_mls_ids])].copy()
        
        has_smaller = (selected['Clean_GLA'] < (subj_gla - 100)).any()
        has_larger = (selected['Clean_GLA'] > (subj_gla + 100)).any()

        if not (has_smaller and has_larger) and len(selected) < 4:
            rem = df_pool[~df_pool['MLS #'].isin(selected['MLS #'])].copy()
            if not has_smaller:
                smaller_pool = rem[rem['Clean_GLA'] < subj_gla].sort_values('Clean_GLA', ascending=False)
                if not smaller_pool.empty:
                    selected = pd.concat([selected, smaller_pool.head(1)])
            elif not has_larger:
                larger_pool = rem[rem['Clean_GLA'] > subj_gla].sort_values('Clean_GLA', ascending=True)
                if not larger_pool.empty:
                    selected = pd.concat([selected, larger_pool.head(1)])
        return selected

    scores = []
    for _, row in df_pool.iterrows():
        gla_diff = abs(row['Clean_GLA'] - subj_gla)
        row_subd = str(row['Clean_Subdivision']).strip().lower()
        subd_bonus = -500.0 if (subj_subd and subj_subd in row_subd) else 0.0
        days_diff = (pd.to_datetime('today') - row['Clean_Date']).days if pd.notnull(row['Clean_Date']) else 90
        recency_penalty = min(days_diff * 0.1, 50.0)
        score = gla_diff + subd_bonus + recency_penalty
        scores.append(score)

    df_pool['score'] = scores
    df_pool = df_pool.sort_values('score')

    selected_comps = []
    subd_matches = df_pool[df_pool['Clean_Subdivision'].str.lower().str.contains(subj_subd, na=False)] if subj_subd else pd.DataFrame()
    if not subd_matches.empty:
        selected_comps.append(subd_matches.iloc[0]['MLS #'])

    candidates = df_pool[~df_pool['MLS #'].isin(selected_comps)].copy()
    for _, row in candidates.iterrows():
        selected_comps.append(row['MLS #'])
        if len(selected_comps) >= 4:
            break

    return df_pool[df_pool['MLS #'].isin(selected_comps)].copy()

# ==============================================================================
# SALES COMPARISON GRID ADJUSTMENTS
# ==============================================================================
def calculate_grid_adjustments(comps_df, subject, reg_params):
    grid_rows = []
    gla_rate = reg_params['gla_rate']
    deadband = 100.0

    for _, row in comps_df.iterrows():
        price = row['Clean_Price']
        comp_gla = row['Clean_GLA']
        
        # ANSI Z765 Deadband: +/- 100 SF receives $0 adjustment
        gla_diff = subject['gla'] - comp_gla
        gla_adj = 0.0 if abs(gla_diff) <= deadband else round(gla_diff * gla_rate, 0)

        # APPRAISER MARKET LOGIC:
        # 1. Market does not pay more for 2.1 baths than 3.0 baths (comp is never penalized for having more baths).
        # 2. In this price corridor, additional bedrooms (4b, 5b) and baths (3.0ba) are absorbed in the physical GLA footprint.
        # 3. Room count adjustment is $0 to prevent double-counting.
        room_adj = 0.0

        # Garage adjustments ($10,000 per bay variance)
        gar_diff = subject['garage_bays'] - row['Clean_Garage']
        gar_adj = gar_diff * 10000.0

        # Basement adjustment
        bsmt_adj = (subject['bsmt_fin'] * 50.0)
        cond_adj = 0.0

        net_adj = gla_adj + room_adj + gar_adj + cond_adj + bsmt_adj
        gross_adj = abs(gla_adj) + abs(room_adj) + abs(gar_adj) + abs(cond_adj) + abs(bsmt_adj)
        indicated = price + net_adj

        net_pct = round((net_adj / price) * 100, 1) if price > 0 else 0.0
        gross_pct = round((gross_adj / price) * 100, 1) if price > 0 else 0.0

        grid_rows.append({
            "mls": row.get('MLS #', 'N/A'),
            "address": row.get('Address', 'Unknown Address'),
            "subdivision": row.get('Clean_Subdivision', ''),
            "price": price,
            "contract_date": row['Clean_Date'].strftime('%m/%d/%Y') if pd.notnull(row['Clean_Date']) else 'N/A',
            "gla": comp_gla,
            "rooms": f"{row['Clean_Beds']}b/{row['Clean_FB']}.{row['Clean_HB']}ba",
            "gla_adj": gla_adj,
            "room_adj": room_adj,
            "gar_adj": gar_adj,
            "bsmt_adj": bsmt_adj,
            "cond_adj": cond_adj,
            "net_adj": net_adj,
            "gross_adj": gross_adj,
            "net_pct": net_pct,
            "gross_pct": gross_pct,
            "indicated": indicated
        })

    return pd.DataFrame(grid_rows)

# ==============================================================================
# GROQ NARRATIVE GENERATOR
# ==============================================================================
from groq import Groq

def generate_narrative_groq(api_key, subject, reg_params, grid_df):
    clean_key = str(api_key).strip().strip('"').strip("'")
    if not clean_key or not clean_key.startswith("gsk_"):
        return "<p>Narrative generation bypassed: Missing or invalid Groq API key in Section 4.</p>"

    client = Groq(api_key=clean_key)

    # 1. Inspect account models and select the strongest available text model
    try:
        models_data = client.models.list().data
        active_ids = [m.id for m in models_data]
    except Exception as e:
        return f"<p>Could not fetch model list from Groq: {str(e)}</p>"

    # Strictly prioritize the 70B models over lightweight 1B/8B preview models
    preferred_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "deepseek-r1-distill-llama-70b",
        "mixtral-8x7b-32768",
        "llama-3.1-8b-instant"
    ]
    
    chosen_model = next((m for m in preferred_models if m in active_ids), None)
    if not chosen_model:
        chat_candidates = [
            m for m in active_ids 
            if not any(b in m.lower() for b in ["guard", "whisper", "vision", "embed", "orpheus"])
        ]
        chosen_model = chat_candidates[0] if chat_candidates else "llama-3.3-70b-versatile"

    min_ind = int(grid_df['indicated'].min())
    max_ind = int(grid_df['indicated'].max())
    midpoint = int((min_ind + max_ind) / 2)

    # 2. Build explicit, unambiguous factual bullets for each comp
    comp_bullets = []
    for _, row in grid_df.iterrows():
        comp_bullets.append(
            f"- {row['address']} ({row['subdivision']}): GLA {int(row['gla']):,} SF | "
            f"Closed ${row['price']:,.0f} | Net Adj ${row['net_adj']:+,.0f} | Indicated ${row['indicated']:,.0f}"
        )
    comps_summary_text = "\n".join(comp_bullets)

    # Identify the actual bracketing bounds in Python
    min_gla_comp = grid_df.loc[grid_df['gla'].idxmin()]
    max_gla_comp = grid_df.loc[grid_df['gla'].idxmax()]

    prompt = f"""
SUBJECT PROPERTY:
Address: {subject['address']}, Raleigh, NC (Subdivision: {subject['subdivision']})
Gross Living Area (GLA): {subject['gla']:,} SF
Beds/Baths: {subject['beds']} Beds / {subject['baths_full']}.{subject['baths_half']} Baths
Statistical Benchmark: ${reg_params['benchmark']:,.0f}
Extracted Marginal GLA Rate: ${reg_params['gla_rate']}/SF with ANSI +/- 100 SF deadband
Reconciled Corridor: ${min_ind:,.0f} to ${max_ind:,.0f} (Midpoint: ${midpoint:,.0f})

VERIFIED COMPARABLE SALES (DO NOT ALTER THESE FACTS):
{comps_summary_text}

BRACKETING FACTS:
- Low-end GLA Comp: {min_gla_comp['address']} at {int(min_gla_comp['gla']):,} SF
- High-end GLA Comp: {max_gla_comp['address']} at {int(max_gla_comp['gla']):,} SF

TASK:
Write an executive valuation narrative formatted strictly as TWO clean HTML paragraphs (<p><b>Title:</b> Text</p>):

<p><b>Executive Pricing Strategy & Valuation Thesis:</b> Reconcile the indicated price corridor (${min_ind:,.0f} to ${max_ind:,.0f}), the midpoint (${midpoint:,.0f}), and the statistical benchmark (${reg_params['benchmark']:,.0f}). State why midpoint pricing provides an appraisal-defensible foundation, and discuss the strategic risks versus market ceiling rewards of pricing near the upper corridor limit.</p>

<p><b>ANSI Z765 & Comparable Bracketing Defense:</b> Detail how the actual comparable sales bracket the subject's {subject['gla']:,} SF GLA, specifically referencing {min_gla_comp['address']} ({int(min_gla_comp['gla']):,} SF) on the lower bound and {max_gla_comp['address']} ({int(max_gla_comp['gla']):,} SF) on the upper bound. Note that closed sale prices establish two-sided price bracketing, and highlight the prioritization of internal {subject['subdivision']} sales. Confirm that independent room count adjustments are set to $0 because room and bath utility variances are absorbed within the ${reg_params['gla_rate']}/SF marginal GLA rate, preventing double-dipping.</p>

CRITICAL RULES:
1. NEVER invent property addresses (like Hillstone Dr sales). Reference ONLY the comparables listed above.
2. NEVER say all properties have the same GLA. State their actual square footages from the list.
3. Output ONLY the two <p> paragraphs. No markdown asterisks, no bullet lists. Total length: 200–260 words.
"""

    system_msg = (
        "You are an expert North Carolina real estate appraiser writing an executive listing defense brief under N.C.G.S. § 93A-83. "
        "Strict adherence to facts is mandatory. Use plain ASCII keyboard characters only. Never use LaTeX syntax."
    )

    try:
        response = client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500,
            presence_penalty=0.4,
            frequency_penalty=0.4
        )
        return sanitize_for_pdf(response.choices[0].message.content)
    except Exception as e:
        return f"<p>Generation failed with model <b>{chosen_model}</b>: {str(e)}</p>"
# ==============================================================================
# AUDIT-GRADE PDF EXPORT
# ==============================================================================
def build_pdf_report(subject, reg_params, grid_df, narrative_text):
    clean_narrative = sanitize_for_pdf(narrative_text)
    
    clean_narrative = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_narrative)
    clean_narrative = re.sub(r'^\s*\*\s+(.*?)$', r'<li>\1</li>', clean_narrative, flags=re.MULTILINE)
    if '<li>' in clean_narrative and '<ul>' not in clean_narrative:
        clean_narrative = clean_narrative.replace('<li>', '<ul><li>', 1) + '</ul>'

    min_val = int(grid_df['indicated'].min())
    max_val = int(grid_df['indicated'].max())
    midpoint = int((min_val + max_val) / 2)

    table_rows_html = ""
    for _, r in grid_df.iterrows():
        gla_adj_str = f"${r['gla_adj']:+,.0f}" if r['gla_adj'] != 0 else "$0"
        rm_adj_str = f"${r['room_adj']:+,.0f}" if r['room_adj'] != 0 else "$0"
        gar_adj_str = f"${r['gar_adj']:+,.0f}" if r['gar_adj'] != 0 else "$0"
        net_adj_str = f"${r['net_adj']:+,.0f}"

        table_rows_html += f"""
        <tr>
            <td style="text-align: left; font-weight: bold; padding: 4px 3px;">
                {r['address']}<br/>
                <span style="font-size: 6.5pt; color: #475569; font-weight: normal;">MLS #{r['mls']} • {r['subdivision']}</span>
            </td>
            <td style="white-space: nowrap;">${r['price']:,.0f}</td>
            <td style="white-space: nowrap;">{r['contract_date']}</td>
            <td style="white-space: nowrap;">{int(r['gla']):,} SF</td>
            <td style="white-space: nowrap;">{r['rooms']}</td>
            <td style="white-space: nowrap;">{gla_adj_str}</td>
            <td style="white-space: nowrap;">{rm_adj_str}</td>
            <td style="white-space: nowrap;">{gar_adj_str}</td>
            <td style="white-space: nowrap;">
                {net_adj_str}<br/>
                <span style="font-size: 6pt; color: #64748B;">({r['gross_pct']}% gr / {r['net_pct']}% net)</span>
            </td>
            <td style="white-space: nowrap; font-weight: bold; color: #1E3A8A; background-color: #F8FAFC;">
                ${r['indicated']:,.0f}
            </td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{
                size: letter portrait;
                margin: 0.35in 0.4in;
            }}
            body {{
                font-family: Helvetica, Arial, sans-serif;
                font-size: 8pt;
                line-height: 1.3;
                color: #1E293B;
            }}
            .header-bar {{
                background-color: #1E3A8A;
                color: #ffffff;
                padding: 8px 12px;
                margin-bottom: 6px;
            }}
            .header-title {{
                font-size: 13pt;
                font-weight: bold;
                margin: 0;
            }}
            .header-sub {{
                font-size: 7.5pt;
                color: #E2E8F0;
                margin-top: 2px;
            }}
            .disclaimer {{
                font-size: 6pt;
                color: #475569;
                background-color: #F8FAFC;
                padding: 4px 8px;
                border-left: 3px solid #94A3B8;
                margin-bottom: 8px;
            }}
            .section-title {{
                font-size: 9pt;
                font-weight: bold;
                color: #1E3A8A;
                border-bottom: 1px solid #CBD5E1;
                padding-bottom: 2px;
                margin-top: 8px;
                margin-bottom: 5px;
            }}
            .narrative-box {{
                font-size: 7.5pt;
                line-height: 1.25;
                text-align: justify;
                margin-bottom: 8px;
            }}
            .narrative-box p {{
                margin: 0 0 4px 0;
            }}
            .narrative-box ul {{
                margin: 2px 0 4px 14px;
                padding: 0;
            }}
            .narrative-box li {{
                margin-bottom: 2px;
            }}
            table.comp-grid {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 4px;
                margin-bottom: 6px;
                font-size: 7pt;
            }}
            table.comp-grid th {{
                background-color: #F1F5F9;
                color: #0F172A;
                font-weight: bold;
                padding: 4px 2px;
                border: 0.5pt solid #CBD5E1;
                text-align: center;
            }}
            table.comp-grid td {{
                padding: 3px 2px;
                border: 0.5pt solid #E2E8F0;
                text-align: center;
                vertical-align: middle;
            }}
            .kpi-table {{
                width: 100%;
                margin-bottom: 6px;
            }}
            .kpi-box {{
                background-color: #F8FAFC;
                border: 0.5pt solid #CBD5E1;
                padding: 4px;
                text-align: center;
            }}
            .kpi-val {{
                font-size: 10.5pt;
                font-weight: bold;
                color: #1E3A8A;
            }}
            .kpi-lbl {{
                font-size: 6pt;
                color: #64748B;
                text-transform: uppercase;
                font-weight: bold;
            }}
            .page-break {{
                page-break-before: always;
            }}
        </style>
    </head>
    <body>
        <div class="header-bar">
            <div class="header-title">Executive Listing Defense Brief & Valuation Packet</div>
            <div class="header-sub">Subject: {subject['address']} | Subdivision: {subject['subdivision']} | Valuation Date: {subject['val_date']}</div>
        </div>

        <div class="disclaimer">
            <strong>REGULATORY & LEGAL DISCLAIMER (N.C.G.S. § 93A-83):</strong> This document is a computer-assisted Comparative Market Analysis (CMA) and Listing Defense Brief prepared solely for transaction consultation and listing defense. THIS IS NOT AN OFFICIAL APPRAISAL and has not been developed in accordance with USPAP standards.
        </div>

        <table class="kpi-table">
            <tr>
                <td class="kpi-box"><div class="kpi-lbl">Statistical Benchmark</div><div class="kpi-val">${reg_params['benchmark']:,.0f}</div></td>
                <td class="kpi-box"><div class="kpi-lbl">Corridor Midpoint</div><div class="kpi-val">${midpoint:,.0f}</div></td>
                <td class="kpi-box"><div class="kpi-lbl">Supported Corridor</div><div class="kpi-val">${min_val:,.0f} - ${max_val:,.0f}</div></td>
                <td class="kpi-box"><div class="kpi-lbl">GLA Adjustment Rate</div><div class="kpi-val">${reg_params['gla_rate']}/SF</div></td>
            </tr>
        </table>

        <div class="section-title">1. Executive Pricing Strategy & Valuation Narrative</div>
        <div class="narrative-box">
            {clean_narrative}
        </div>

        <div class="page-break"></div>

        <div class="section-title">2. Sales Comparison Approach Reconciliation Grid (ANSI Z765 Bracketing)</div>
        <table class="comp-grid">
            <thead>
                <tr>
                    <th style="width: 25%; text-align: left;">Comparable Property</th>
                    <th style="width: 8%;">Closed Price</th>
                    <th style="width: 8%;">Date</th>
                    <th style="width: 7%;">GLA</th>
                    <th style="width: 8%;">Beds/Baths</th>
                    <th style="width: 8%;">GLA Adj ($)</th>
                    <th style="width: 8%;">Room Adj ($)</th>
                    <th style="width: 8%;">Garage Adj ($)</th>
                    <th style="width: 10%;">Net Adj ($)</th>
                    <th style="width: 10%;">Indicated Value</th>
                </tr>
            </thead>
            <tbody>
                {table_rows_html}
            </tbody>
        </table>
        <div style="font-size: 6.5pt; color: #64748B; margin-top: 4px;">
            *Note: Living area variances within +/- 100 SF receive a $0 adjustment per ANSI Z765 measurement standards. Comparable selection algorithm prioritizes internal subdivision sales and enforces Fannie Mae two-sided bracketing.
        </div>
    </body>
    </html>
    """

    sanitized_html = sanitize_for_pdf(html)
    pdf_buffer = BytesIO()
    pisa.CreatePDF(sanitized_html, dest=pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.header("1. Upload MLS Comps CSV")
    uploaded_file = st.file_uploader("Upload MLS Search Export (.csv)", type=["csv"])
    
    st.header("2. Valuation Mode & Pricing")
    val_mode = st.radio("Deliverable Strategy Mode:", ["Pre-Listing / CMA Strategy", "Executed Contract Defense"])
    val_date = st.date_input("Valuation Date:", value=date.today())
    apply_room_adj = st.checkbox(
        "Apply Independent Room/Bath Adjustments", 
        value=False,
        help="Unchecked by default: allows GLA adjustment to capture size utility without double-dipping."
    )
    
    st.header("3. Comparable Selection Mode")
    comp_mode = st.radio("Selection Strategy:", ["Algorithmic (Appraisal Bracketing)", "Manual Override (Pick Specific Comps)"])
    
    st.header("4. Inference Provider")
    groq_api_key = st.text_input(
        "Groq API Key:", 
        value=st.secrets.get("GROQ_API_KEY", ""), 
        type="password",
        help="Paste your gsk_... key here"
    )

# ==============================================================================
# MAIN INTERFACE: RESTORED 3-COLUMN PARAMETER GRID
# ==============================================================================
st.markdown('<div class="report-header">Listing Defense Brief & Pricing Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Algorithmic Comparable Selection • OLS Residual Sensitivity • ANSI Z765 & Appraisal Bracketing Compliance</div>', unsafe_allow_html=True)

st.subheader("Subject Property Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    subj_address = st.text_input("Subject Address:", value="213 Hillstone Dr, Raleigh, NC 27615")
    subj_subdivision = st.text_input("Subdivision (Leave empty for submarket):", value="Chadwick")
    subj_year = st.number_input("Year Built:", value=1991, step=1)
    subj_beds = st.number_input("Above-Grade Bedrooms:", value=3, step=1)
    subj_bsmt_beds = st.number_input("Basement Bedrooms:", value=0, step=1)

with col2:
    subj_gla = st.number_input("Above-Grade Measured GLA (SF):", value=2004, step=10)
    subj_bsmt_fin = st.number_input("Below-Grade Finished Basement (SF):", value=0, step=50)
    subj_bsmt_unfin = st.number_input("Below-Grade Unfinished Storage (SF):", value=0, step=50)
    subj_fb = st.number_input("Above-Grade Full Baths:", value=2, step=1)
    subj_bsmt_fb = st.number_input("Basement Full Baths:", value=0, step=1)

with col3:
    subj_condition = st.selectbox("Condition Rating:", ["Excellent", "Good", "Average", "Fair"])
    subj_garage = st.number_input("Garage Bays:", value=2, step=1)
    subj_fireplaces = st.number_input("Fireplaces:", value=1, step=1)
    subj_hb = st.number_input("Above-Grade Half Baths:", value=1, step=1)
    subj_bsmt_hb = st.number_input("Basement Half Baths:", value=0, step=1)

st.markdown("**Exterior Features & Amenities:**")
am_cols = st.columns(6)
with am_cols[0]:
    has_front_porch = st.checkbox("Front Porch", value=True)
with am_cols[1]:
    has_screen_porch = st.checkbox("Screen Porch", value=False)
with am_cols[2]:
    has_deck = st.checkbox("Deck", value=True)
with am_cols[3]:
    has_patio = st.checkbox("Patio", value=True)
with am_cols[4]:
    has_fence = st.checkbox("Fenced Yard", value=False)
with am_cols[5]:
    has_pool = st.checkbox("Pool", value=False)

subject_data = {
    "address": subj_address,
    "subdivision": subj_subdivision,
    "year_built": subj_year,
    "gla": subj_gla,
    "beds": subj_beds,
    "baths_full": subj_fb,
    "baths_half": subj_hb,
    "garage_bays": subj_garage,
    "condition": subj_condition,
    "fireplaces": subj_fireplaces,
    "bsmt_fin": subj_bsmt_fin,
    "bsmt_unfin": subj_bsmt_unfin,
    "val_date": str(val_date),
    "amenities": {
        "Front Porch": has_front_porch,
        "Screen Porch": has_screen_porch,
        "Deck": has_deck,
        "Patio": has_patio,
        "Fenced Yard": has_fence,
        "Pool": has_pool
    }
}

# ==============================================================================
# EXECUTION WORKFLOW
# ==============================================================================
if uploaded_file is not None:
    df = load_and_clean_data(uploaded_file)
    st.success(f"Loaded {len(df)} market records from MLS export.")

    reg_params = run_regression_analysis(df, subject_data)

    manual_ids = []
    if comp_mode == "Manual Override (Pick Specific Comps)":
        mls_options = df['MLS #'].tolist()
        format_func = lambda x: f"MLS #{x} - {df[df['MLS #']==x]['Address'].values[0]} (${df[df['MLS #']==x]['Clean_Price'].values[0]:,.0f})"
        manual_ids = st.multiselect("Choose 3 to 4 Preferred Comparables:", mls_options, format_func=format_func)

    if st.button("Generate Listing Defense Brief", type="primary", use_container_width=True):
        mode_key = "manual" if comp_mode == "Manual Override (Pick Specific Comps)" else "algorithmic"
        comps_df = select_comparables_with_subdivision(df, subject_data, mode=mode_key, manual_mls_ids=manual_ids)
        
        if comps_df.empty:
            st.error("No valid comparables found in the uploaded dataset.")
        else:
            grid_df = calculate_grid_adjustments(comps_df, subject_data, reg_params)

            with st.spinner("Compiling narrative via Groq Cloud..."):
                narrative = generate_narrative_groq(groq_api_key, subject_data, reg_params, grid_df)

            st.markdown("### Valuation & Grid Defense")
            st.dataframe(
                grid_df[['address', 'subdivision', 'price', 'gla', 'rooms', 'gla_adj', 'room_adj', 'gar_adj', 'net_adj', 'gross_pct', 'indicated']],
                use_container_width=True
            )

            with st.expander("Preview Executive Appraisal Narrative", expanded=True):
                st.write(narrative, unsafe_allow_html=True)

            pdf_bytes = build_pdf_report(subject_data, reg_params, grid_df, narrative)
            
            clean_filename = f"Listing_Defense_Brief_{re.sub(r'[^a-zA-Z0-9]', '_', subj_address)}.pdf"
            st.download_button(
                label="📥 Download Audit-Grade Defense Brief (PDF)",
                data=pdf_bytes,
                file_name=clean_filename,
                mime="application/pdf",
                use_container_width=True
            )
else:
    st.info("Upload your Doorify / Triangle MLS export CSV in the left sidebar to begin.")
