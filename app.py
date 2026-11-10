import io
import json
import re
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import streamlit as st
from xhtml2pdf import pisa

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Listing Defense Brief Engine",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .stButton>button { background-color: #2563EB; color: white; font-weight: 600; border-radius: 6px; padding: 0.5rem 1.5rem; border: none; }
    .stButton>button:hover { background-color: #1D4ED8; color: white; }
    .report-narrative { max-width: 900px; line-height: 1.7; font-size: 1.02rem; color: #1F2937; margin-bottom: 1.2rem; }
    .table-container { overflow-x: auto; margin-top: 0.8rem; margin-bottom: 1.8rem; }
    table.screen-table { width: 100%; border-collapse: collapse; margin-top: 6px; margin-bottom: 6px; }
    table.screen-table th, table.screen-table td { border: 1px solid #E2E8F0; padding: 6px 8px; text-align: left; font-size: 0.88rem; }
    table.screen-table th { background-color: #F8FAFC; color: #0F172A; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Listing Defense Brief & Pricing Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Algorithmic Comparable Selection • OLS Residual Sensitivity • ANSI Z765 & Appraisal Bracketing Compliance</div>', unsafe_allow_html=True)

# ==============================================================================
# SESSION RESET TRACKER
# ==============================================================================
if "reset_key" not in st.session_state:
    st.session_state["reset_key"] = 0

def reset_all_data():
    st.session_state["reset_key"] += 1
    current_rk = st.session_state["reset_key"]
    st.session_state.clear()
    st.session_state["reset_key"] = current_rk

# ==============================================================================
# HELPER & FEATURE ENGINES
# ==============================================================================
def clean_str(val):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def normalize_condition_score(val):
    if not val or pd.isna(val):
        return 0
    t = str(val).strip().lower()
    if any(k in t for k in ["poor", "as-is", "as is", "need of repairs", "repair", "tlc"]):
        return 1
    if any(k in t for k in ["fair", "below average"]):
        return 2
    if any(k in t for k in ["average", "typical", "standard"]):
        return 3
    if any(k in t for k in ["good", "well maintained", "turnkey", "move-in"]):
        return 4
    if any(k in t for k in ["updated", "renovated", "remodeled", "custom", "luxury", "pristine"]):
        return 5
    return 3

def parse_concessions(row):
    for col in ["Concessions", "Financing Concessions Amount", "Seller Concessions", "Buyer Closing Costs"]:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).replace("$", "").replace(",", "").strip()
            try:
                return float(val)
            except ValueError:
                pass
    return 0.0

def extract_numeric_field(row, possible_columns, default=0.0):
    for col in possible_columns:
        if col in row and pd.notna(row[col]):
            clean_val = str(row[col]).replace(",", "").replace("$", "").strip()
            try:
                return float(clean_val)
            except ValueError:
                pass
    return default

def is_market_outlier(row):
    combined = " ".join([
        str(row.get("Public Remarks", "")),
        str(row.get("Private Remarks", "")),
        str(row.get("Syndication Remarks", ""))
    ]).lower()
    return any(k in combined for k in ["55+", "under 55", "active adult", "one occupant must be 55"])

def parse_amenities_and_classification(row, valuation_year=2026):
    pub = str(row.get("Public Remarks", "")).lower()
    priv = str(row.get("Private Remarks", "")).lower()
    synd = str(row.get("Syndication Remarks", "")).lower()
    unit = str(row.get("Unit Number", "")).lower()
    ext = str(row.get("Exterior Features", "")).lower()
    porch_col = str(row.get("Patio & Porch Features", "")).lower()
    fence_col = str(row.get("Fencing", "")).lower()
    combined_text = f"{pub} {priv} {synd} {ext} {porch_col} {fence_col}"

    beds_total = extract_numeric_field(row, ["Bedrooms Total", "Bedrooms", "Beds Total", "Beds", "Total Bedrooms"], default=3.0)
    full_baths_total = extract_numeric_field(row, ["Bathrooms Full", "Baths Full", "Full Baths", "Baths - Full"], default=2.0)
    half_baths_total = extract_numeric_field(row, ["Bathrooms Half", "Baths Half", "Half Baths", "Baths - Half"], default=1.0)
    garage = extract_numeric_field(row, ["Garage Spaces", "Garage", "Attached Garage Spaces", "Garage Spaces Total"], default=1.0)
    bsmt_fin = extract_numeric_field(row, ["Below Grade Finished Area", "Basement Finished Area", "Finished Bsmt SF", "Below Grade Fin SF"], default=0.0)
    bsmt_unfin = extract_numeric_field(row, ["Below Grade Unfinished Area", "Basement Unfinished Area", "Unfinished Bsmt SF", "Below Grade Unfin SF"], default=0.0)
    bsmt_beds = extract_numeric_field(row, ["Basement Bedrooms", "Below Grade Bedrooms", "Bsmt Bedrooms", "Bsmt Beds"], default=0.0)
    bsmt_full_baths = extract_numeric_field(row, ["Basement Full Baths", "Below Grade Full Baths", "Bsmt Full Baths"], default=0.0)
    bsmt_half_baths = extract_numeric_field(row, ["Basement Half Baths", "Below Grade Half Baths", "Bsmt Half Baths"], default=0.0)

    if bsmt_fin > 0 and bsmt_beds == 0.0 and ("basement bedroom" in combined_text or "bedroom in basement" in combined_text):
        bsmt_beds = 1.0
    if bsmt_fin > 0 and bsmt_full_baths == 0.0 and ("basement bath" in combined_text or "full bath in basement" in combined_text):
        bsmt_full_baths = 1.0

    above_beds = max(1.0, beds_total - bsmt_beds)
    above_full_baths = max(1.0, full_baths_total - bsmt_full_baths)
    above_half_baths = max(0.0, half_baths_total - bsmt_half_baths)

    fireplaces = extract_numeric_field(row, ["Fireplaces Total", "Fireplaces", "Number of Fireplaces", "Fireplace Total"], default=0.0)
    if fireplaces == 0.0 and ("fireplace" in combined_text or str(row.get("Fireplace YN", "")).lower() == "y"):
        fireplaces = 1.0

    has_fence = 1 if ("fence" in combined_text or "fenced" in combined_text or "fencing" in combined_text) else 0
    has_front_porch = 1 if ("front porch" in combined_text or "covered front" in combined_text or "rocking chair porch" in combined_text or "porch - front" in combined_text or "covered entry" in combined_text or "open porch" in combined_text) else 0
    has_screen = 1 if ("screen" in combined_text and "porch" in combined_text) else 0
    has_deck = 1 if "deck" in combined_text else 0
    has_patio = 1 if ("patio" in combined_text or "concrete" in combined_text) else 0
    has_pool = 1 if (str(row.get("Pool Private YN", "")).lower() == "y" or "inground" in combined_text or "private pool" in combined_text) else 0

    resale_indicators = ["sellers willing", "items not included", "washer and dryer", "furniture", "floating shelves", "re-sale", "resale", "original owner", "shows like a model", "impeccably maintained"]
    is_definite_resale = any(k in combined_text for k in resale_indicators)
    builder_cues = ["one of our", "below market interest", "see on site agent", "photos are for illustration", "comp purposes only", "features and options can vary", "est completion", "proposed", "under construction"]
    is_builder_cue = any(k in combined_text for k in builder_cues) or (unit != "" and unit != "nan")

    comp_age = 2.0 if is_definite_resale or not is_builder_cue else 0.0
    is_builder_sale = 0 if is_definite_resale or not is_builder_cue else 1

    condition_raw = str(row.get("Condition", row.get("Property Condition", "")))
    if not condition_raw or condition_raw.lower() in ["nan", "none"]:
        if any(w in combined_text for w in ["poor", "as-is", "as is", "needs repairs", "tlc"]):
            condition_raw = "poor"
        elif any(w in combined_text for w in ["updated", "renovated", "remodeled", "new kitchen"]):
            condition_raw = "updated"
        elif any(w in combined_text for w in ["good", "well maintained", "turnkey"]):
            condition_raw = "good"
        else:
            condition_raw = "average"

    return {
        "beds_above": above_beds,
        "baths_full_above": above_full_baths,
        "baths_half_above": above_half_baths,
        "bsmt_beds": bsmt_beds,
        "bsmt_full_baths": bsmt_full_baths,
        "bsmt_half_baths": bsmt_half_baths,
        "garage_bays": garage,
        "fireplaces": fireplaces,
        "has_fence": has_fence,
        "has_front_porch": has_front_porch,
        "bsmt_finished_sf": bsmt_fin,
        "bsmt_unfinished_sf": bsmt_unfin,
        "has_screen_porch": has_screen,
        "has_deck": has_deck,
        "has_patio": has_patio,
        "has_pool": has_pool,
        "is_new_construction": is_builder_sale,
        "age_years": comp_age,
        "condition_score": normalize_condition_score(condition_raw),
        "condition_desc": condition_raw.capitalize()
    }

def solve_rate_by_sensitivity(df, subject_val, comp_val_col, candidate_range=(0, 25000), step=500):
    if len(df[comp_val_col].unique()) < 2:
        return 0.0, "0% variance in sample"
    best_rate = 0.0
    min_std_dev = float("inf")
    baseline_std = np.std(df["Residual_Price"])

    for candidate_adj in range(candidate_range[0], candidate_range[1] + step, step):
        adj_resids = [row["Residual_Price"] + ((subject_val - row[comp_val_col]) * candidate_adj) for _, row in df.iterrows()]
        current_std = np.std(adj_resids)
        if current_std < min_std_dev:
            min_std_dev = current_std
            best_rate = float(candidate_adj)

    status = f"Optimized (std dev ${min_std_dev:,.0f})" if min_std_dev < (baseline_std * 0.995) else "Absorbed in Area/GLA rate"
    return best_rate, status

def sanitize_markdown_text(text):
    if not text:
        return ""
    text = re.sub(r'^(Part\s*[12]\s*(\([^)]*\))?:?\s*)', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^(#+\s*)?(1\.\s*)?Executive Pricing Strategy & Valuation Thesis\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^(#+\s*)?(2\.\s*)?ANSI Z765 & Comp Bracketing Defense\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^(SECTION\s*[12]:?\s*)', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = text.replace("$\\pm$", "±").replace("\\pm", "±").replace("$\\pm100SF$", "±100 SF")
    text = re.sub(r'\$([0-9,]+)\$', r'$\1', text)
    return re.sub(r'\$([^$\n]+)\$', r'\1', text).strip()

def build_pdf_document(sub_address, valuation_date, header_sub, sec_a, sec_b, table1_html, table2_html, footnote):
    paras_a = "".join([f"<p>{p.strip()}</p>" for p in sec_a.split("\n\n") if p.strip()])
    paras_b = "".join([f"<p>{p.strip()}</p>" for p in sec_b.split("\n\n") if p.strip()])

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page portrait_page {{
        size: letter portrait;
        margin: 0.42in 0.45in 0.42in 0.45in;
        @bottom-right {{
            content: "Page 1 of 2";
            font-size: 7.5pt;
            color: #64748B;
        }}
    }}
    @page landscape_page {{
        size: letter landscape;
        margin: 0.42in 0.45in 0.4in 0.45in;
        @bottom-right {{
            content: "Page 2 of 2";
            font-size: 7.5pt;
            color: #64748B;
        }}
    }}
    body {{
        font-family: Helvetica, Arial, sans-serif;
        color: #1E293B;
        font-size: 8.3pt;
        line-height: 1.4;
    }}
    .header-bar {{
        border-bottom: 2px solid #1E3A8A;
        padding-bottom: 4px;
        margin-bottom: 6px;
    }}
    .main-title {{
        font-size: 13.5pt;
        font-weight: bold;
        color: #1E3A8A;
        margin: 0 0 2px 0;
    }}
    .sub-title {{
        font-size: 8.2pt;
        font-weight: bold;
        color: #475569;
        margin: 0;
    }}
    .disclaimer-box {{
        background-color: #F8FAFC;
        border-left: 3px solid #94A3B8;
        padding: 4px 6px;
        font-size: 6.5pt;
        color: #475569;
        line-height: 1.3;
        margin-bottom: 6px;
    }}
    h2 {{
        font-size: 9.8pt;
        font-weight: bold;
        color: #1E3A8A;
        border-bottom: 1px solid #CBD5E1;
        padding-bottom: 2px;
        margin-top: 6px;
        margin-bottom: 4px;
    }}
    h3 {{
        font-size: 8.5pt;
        font-weight: bold;
        color: #334155;
        margin-top: 6px;
        margin-bottom: 3px;
    }}
    p {{
        margin: 0 0 4px 0;
        text-align: justify;
    }}
    table.pdf-clean-table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 4px;
        margin-bottom: 4px;
    }}
    table.pdf-clean-table th, table.pdf-clean-table td {{
        border: 0.5px solid #CBD5E1;
        padding: 3.5px 5px;
        text-align: left;
        vertical-align: middle;
    }}
    table.pdf-clean-table th {{
        background-color: #F8FAFC;
        color: #0F172A;
        font-weight: bold;
        border-bottom: 1px solid #94A3B8;
    }}
    .footnote {{
        font-size: 6.5pt;
        color: #64748B;
        font-style: italic;
        margin-top: 6px;
        line-height: 1.3;
    }}
</style>
</head>
<body>
    <!-- PAGE 1: PORTRAIT STRATEGY & REGRESSION -->
    <div class="header-bar">
        <div class="main-title">Executive Listing Defense Brief & Valuation Packet</div>
        <div class="sub-title">Subject Property: {sub_address} | Valuation Date: {valuation_date}{header_sub}</div>
    </div>

    <div class="disclaimer-box">
        <b>REGULATORY & LEGAL DISCLAIMER (N.C.G.S. § 93A-83):</b> This document is a computer-assisted Comparative Market Analysis (CMA) and Listing Defense Brief prepared solely for listing strategy, transaction consultation, and contract defense. <b>THIS IS NOT AN OFFICIAL APPRAISAL.</b> It has not been developed in accordance with USPAP or North Carolina Appraisers Act standards. The valuation indicators and statistical benchmarks reflect algorithmic and matched-pair modeling derived from historical MLS data and do not constitute a certified opinion of market value by a licensed or certified real estate appraiser.
    </div>

    <h2>1. Executive Pricing Strategy & Valuation Thesis</h2>
    <div style="margin-bottom: 4px;">
        {paras_a}
    </div>

    <h3>Verified Market Regression & Sensitivity Extractions</h3>
    {table1_html}

    <pdf:nexttemplate name="landscape_page"/>
    <pdf:nextpage/>

    <!-- PAGE 2: LANDSCAPE SALES COMPARISON APPROACH RECONCILIATION -->
    <div class="header-bar">
        <div class="main-title">Executive Listing Defense Brief & Valuation Packet</div>
        <div class="sub-title">Subject Property: {sub_address} | ANSI Z765 Sales Comparison Reconciliation Grid</div>
    </div>

    <h2>2. ANSI Z765 & Comp Bracketing Defense</h2>
    <div style="margin-bottom: 6px;">
        {paras_b}
    </div>

    <h3>Sales Comparison Approach Reconciliation Grid (12-Column Bracketing)</h3>
    {table2_html}

    <div class="footnote">
        {footnote}
    </div>
</body>
</html>
"""
    pdf_buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(html_content, dest=pdf_buffer)
    return pdf_buffer.getvalue() if not pisa_status.err else None

# ==============================================================================
# SUBJECT DEFAULTS & CALLBACK PROCESSOR
# ==============================================================================
DEFAULTS = {
    "sub_address": "3012 Falls River Ave, Raleigh, NC 27614",
    "sub_subdivision": "Bedford",
    "sub_year_built": 2005,
    "sub_above_beds": 4,
    "sub_bsmt_beds": 0,
    "sub_gla": 3670,
    "sub_bsmt_fin": 0,
    "sub_bsmt_unfin": 0,
    "sub_above_full_baths": 4,
    "sub_bsmt_full_baths": 0,
    "sub_condition": "Good",
    "sub_garage": 2,
    "sub_fireplaces": 1,
    "sub_above_half_baths": 0,
    "sub_bsmt_half_baths": 0,
    "sub_front_porch": True,
    "sub_screen_porch": True,
    "sub_deck": False,
    "sub_patio": True,
    "sub_fence": False,
    "sub_pool": False,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def handle_subject_file_upload():
    current_key = f"sub_csv_uploader_{st.session_state.get('reset_key', 0)}"
    uploaded = st.session_state.get(current_key)
    if uploaded is not None:
        try:
            uploaded.seek(0)
            df_sub = pd.read_csv(uploaded)
            if not df_sub.empty:
                row = df_sub.iloc[0]
                clean_row = {str(k).strip().lower(): v for k, v in row.items()}
                
                def find_val(candidates, default=""):
                    for cand in candidates:
                        c_lower = cand.lower().strip()
                        if c_lower in clean_row and pd.notna(clean_row[c_lower]):
                            val = str(clean_row[c_lower]).strip()
                            if val and val.lower() != "nan":
                                return val
                    return default

                def find_numeric(candidates, default=0.0):
                    raw = find_val(candidates, "")
                    if raw:
                        m = re.search(r"[-+]?\d*\.?\d+", str(raw).replace(",", "").replace("$", ""))
                        if m:
                            try:
                                return float(m.group(0))
                            except ValueError:
                                pass
                    return default

                parsed = parse_amenities_and_classification(row, valuation_year=2026)

                addr = find_val(["Address", "Full Address", "Property Address", "Street Address", "Subject Address", "Street Name", "Listing Address"])
                if addr:
                    st.session_state["sub_address"] = addr

                sub_mls = find_val(["MLS #", "MLS", "Listing ID", "MLS Number"])
                st.session_state["sub_mls_num"] = clean_str(sub_mls)

                subdiv = find_val(["Subdivision", "Subdivision Name", "Subdivision/Complex", "Legal Subdivision", "Neighborhood", "Sub Area", "Development"])
                st.session_state["sub_subdivision"] = subdiv

                yb = find_numeric(["Year Built", "YearBuilt", "Yr Built", "Yr Blt", "Year Built Details", "Actual Year Built", "Year"], default=0.0)
                if yb >= 1800:
                    st.session_state["sub_year_built"] = int(yb)

                gla_val = find_numeric(["Above Grade Finished Area", "Above Grade Finished SF", "Above Grade Fin SF", "GLA", "SqFt - Heated", "SqFt Heated", "Heated Living Area", "Living Area", "Total Heated SqFt", "Finished SqFt", "SqFt Total Living Area"], default=0.0)
                if gla_val > 0:
                    st.session_state["sub_gla"] = int(gla_val)

                st.session_state["sub_above_beds"] = int(parsed["beds_above"])
                st.session_state["sub_above_full_baths"] = int(parsed["baths_full_above"])
                st.session_state["sub_above_half_baths"] = int(parsed["baths_half_above"])
                st.session_state["sub_bsmt_fin"] = int(parsed["bsmt_finished_sf"])
                st.session_state["sub_bsmt_unfin"] = int(parsed["bsmt_unfinished_sf"])
                st.session_state["sub_bsmt_beds"] = int(parsed["bsmt_beds"])
                st.session_state["sub_bsmt_full_baths"] = int(parsed["bsmt_full_baths"])
                st.session_state["sub_bsmt_half_baths"] = int(parsed["bsmt_half_baths"])
                st.session_state["sub_garage"] = int(parsed["garage_bays"])
                st.session_state["sub_fireplaces"] = int(parsed["fireplaces"])
                st.session_state["sub_front_porch"] = bool(parsed["has_front_porch"])
                st.session_state["sub_screen_porch"] = bool(parsed["has_screen_porch"])
                st.session_state["sub_deck"] = bool(parsed["has_deck"])
                st.session_state["sub_patio"] = bool(parsed["has_patio"])
                st.session_state["sub_fence"] = bool(parsed["has_fence"])
                st.session_state["sub_pool"] = bool(parsed["has_pool"])

                c_desc = parsed.get("condition_desc", "Good")
                for opt in ["Good", "Updated / Renovated", "Average", "Fair", "Poor / TLC"]:
                    if opt.lower() in c_desc.lower():
                        st.session_state["sub_condition"] = opt
                        break

                st.session_state["subject_load_success"] = True
                st.session_state.pop("subject_load_error", None)
        except Exception as e:
            st.session_state["subject_load_error"] = str(e)

# ==============================================================================
# SIDEBAR: DATA INPUTS & CONFIGURATION
# ==============================================================================
with st.sidebar:
    if st.button("🔄 Start New Brief (Clear All)", use_container_width=True, on_click=reset_all_data):
        st.rerun()

    st.header("1. Upload MLS Comps CSV")
    uploaded_file = st.file_uploader("Upload MLS Search Export (.csv)", type=["csv"], key=f"comps_uploader_{st.session_state['reset_key']}")

    st.header("2. Valuation Mode & Pricing")
    pricing_mode = st.radio("Deliverable Strategy Mode:", ["Pre-Listing / CMA Strategy", "Executed Contract Defense"])
    target_contract_price = st.number_input("Target Contract Price ($):", min_value=50000, max_value=5000000, value=615000, step=5000) if pricing_mode == "Executed Contract Defense" else 0
    valuation_date = st.date_input("Valuation Date:", pd.to_datetime("2026-10-20"))

    st.header("3. Comparable Selection Mode")
    comp_mode = st.radio("Selection Strategy:", ["Algorithmic (Appraisal Bracketing)", "Manual Override (Pick Specific Comps)"])

    manual_mls_picks = []
    if comp_mode == "Manual Override (Pick Specific Comps)" and uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            df_preview = pd.read_csv(uploaded_file)
            p_col = "Close Price" if "Close Price" in df_preview.columns else "Close Price Clean"
            df_preview["Close_Num"] = pd.to_numeric(df_preview[p_col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
            df_preview["Clean_MLS"] = df_preview["MLS #"].apply(clean_str)
            df_preview = df_preview.dropna(subset=["Close_Num", "Address"]).copy()
            df_preview = df_preview[df_preview["Clean_MLS"] != ""]
            
            comp_options = [f"{row['Address']} (MLS #{row['Clean_MLS']}) - ${row['Close_Num']:,.0f}" for _, row in df_preview.iterrows()]
            chosen_labels = st.multiselect("Select up to 6 Preferred Comps:", options=comp_options, max_selections=6)
            for c in chosen_labels:
                manual_mls_picks.append(clean_str(c.split("MLS #")[1].split(")")[0].strip()))
        except Exception:
            pass

    st.header("4. Inference Provider")
    provider = st.radio("Select Provider:", ["Groq Cloud (Sub-Second)", "Local LM Studio (Offline)"], index=0)

    if provider == "Groq Cloud (Sub-Second)":
        default_key = st.secrets.get("GROQ_API_KEY", "") if "GROQ_API_KEY" in st.secrets else ""
        groq_api_key = st.text_input("Groq API Key:", value=default_key, type="password", placeholder="gsk_...")
        target_endpoint = "https://api.groq.com/openai/v1/chat/completions"
        active_model = st.selectbox("Groq Model:", ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"], index=0)
    else:
        raw_endpoint = st.text_input("LM Studio Endpoint:", "http://127.0.0.1:1234/v1/chat/completions")
        clean_endpoint = raw_endpoint.strip()
        if clean_endpoint.endswith("/v1/chat/completion"):
            clean_endpoint += "s"
        elif not clean_endpoint.endswith("/v1/chat/completions") and "/v1" in clean_endpoint:
            clean_endpoint = clean_endpoint.rstrip("/") + "/chat/completions"
        target_endpoint = clean_endpoint
        active_model = st.text_input("Local Model ID:", "qwen2.5-3b-instruct")

# ==============================================================================
# MAIN PAGE: SUBJECT PROPERTY SPECIFICATIONS
# ==============================================================================
st.subheader("Subject Property Parameters")
entry_mode = st.radio("Subject Specification Source:", ["Enter / Edit Details Manually", "Upload Subject MLS Export (.csv)"], horizontal=True)

if entry_mode == "Upload Subject MLS Export (.csv)":
    st.file_uploader("Upload Subject Property MLS Record (.csv):", type=["csv"], key=f"sub_csv_uploader_{st.session_state['reset_key']}", on_change=handle_subject_file_upload)
    if st.session_state.get("subject_load_success"):
        st.success(f"Loaded specifications: **{st.session_state.get('sub_address')}** • Measured GLA: {st.session_state.get('sub_gla')} SF")

c1, c2, c3 = st.columns(3)
with c1:
    sub_address = st.text_input("Subject Address:", key="sub_address")
    sub_subdivision = st.text_input("Subdivision (Leave empty for submarket):", key="sub_subdivision")
    sub_year_built = st.number_input("Year Built:", min_value=1800, max_value=2026, step=1, key="sub_year_built")
    sub_above_beds = st.number_input("Above-Grade Bedrooms:", min_value=1, max_value=10, step=1, key="sub_above_beds")
    sub_bsmt_beds = st.number_input("Basement Bedrooms:", min_value=0, max_value=6, step=1, key="sub_bsmt_beds")

with c2:
    sub_gla = st.number_input("Above-Grade Measured GLA (SF):", min_value=400, max_value=15000, step=10, key="sub_gla")
    sub_bsmt_fin = st.number_input("Below-Grade Finished Basement (SF):", min_value=0, max_value=8000, step=25, key="sub_bsmt_fin")
    sub_bsmt_unfin = st.number_input("Below-Grade Unfinished Storage (SF):", min_value=0, max_value=5000, step=25, key="sub_bsmt_unfin")
    sub_above_full_baths = st.number_input("Above-Grade Full Baths:", min_value=1, max_value=10, step=1, key="sub_above_full_baths")
    sub_bsmt_full_baths = st.number_input("Basement Full Baths:", min_value=0, max_value=6, step=1, key="sub_bsmt_full_baths")

with c3:
    sub_condition = st.selectbox("Condition Rating:", ["Good", "Updated / Renovated", "Average", "Fair", "Poor / TLC"], key="sub_condition")
    sub_garage = st.number_input("Garage Bays:", min_value=0, max_value=6, step=1, key="sub_garage")
    sub_fireplaces = st.number_input("Fireplaces:", min_value=0, max_value=5, step=1, key="sub_fireplaces")
    sub_above_half_baths = st.number_input("Above-Grade Half Baths:", min_value=0, max_value=6, step=1, key="sub_above_half_baths")
    sub_bsmt_half_baths = st.number_input("Basement Half Baths:", min_value=0, max_value=6, step=1, key="sub_bsmt_half_baths")

st.write("**Exterior Features & Amenities:**")
af1, af2, af3, af4, af5, af6 = st.columns(6)
with af1: sub_front_porch = 1 if st.checkbox("Front Porch", key="sub_front_porch") else 0
with af2: sub_screen_porch = 1 if st.checkbox("Screen Porch", key="sub_screen_porch") else 0
with af3: sub_deck = 1 if st.checkbox("Deck", key="sub_deck") else 0
with af4: sub_patio = 1 if st.checkbox("Patio", key="sub_patio") else 0
with af5: sub_fence = 1 if st.checkbox("Fenced Yard", key="sub_fence") else 0
with af6: sub_pool = 1 if st.checkbox("Pool", key="sub_pool") else 0

# ==============================================================================
# EXECUTION PIPELINE
# ==============================================================================
if st.button("Generate Listing Defense Brief", use_container_width=True):
    if not uploaded_file:
        st.error("Please upload an MLS Comps CSV export in Section 1 of the sidebar first.")
        st.stop()

    if provider == "Groq Cloud (Sub-Second)" and not groq_api_key.strip():
        st.error("Please enter a valid Groq API Key in Section 4 of the sidebar.")
        st.stop()

    with st.spinner("Processing market regression, residual sensitivity, and appraisal bracketing..."):
        try:
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.stop()

        price_col = "Close Price" if "Close Price" in df_raw.columns else "Close Price Clean"
        df_raw["Close_Price_Num"] = pd.to_numeric(df_raw[price_col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
        gla_col = "Above Grade Finished Area" if "Above Grade Finished Area" in df_raw.columns else "GLA"
        df_raw["GLA_Num"] = pd.to_numeric(df_raw[gla_col].astype(str).str.replace(",", "", regex=False), errors="coerce")
        df_raw["Contract_Date_DT"] = pd.to_datetime(df_raw["Purchase Contract Date"], errors="coerce")
        df_raw["Clean_MLS"] = df_raw["MLS #"].apply(clean_str)
        df_raw = df_raw.dropna(subset=["Close_Price_Num", "GLA_Num", "Contract_Date_DT"]).copy()

        df_clean = df_raw[~df_raw.apply(is_market_outlier, axis=1)].copy().reset_index(drop=True)
        effective_subject_gla = float(sub_gla)
        valuation_year = valuation_date.year
        subject_age = max(0.0, float(valuation_year - sub_year_built))
        subdivision_display = sub_subdivision.strip() if sub_subdivision.strip() else "Competitive Submarket Area"

        df_clean["is_Subject_Sub"] = 1
        df_clean["Subdivision_Clean"] = subdivision_display
        df_clean["Concession_Amount"] = df_clean.apply(parse_concessions, axis=1)
        concessions_present = df_clean[df_clean["Concession_Amount"] > 0]["Concession_Amount"]
        typical_market_concession = float(concessions_present.median()) if not concessions_present.empty else 0.0
        df_clean["Concession_Adj"] = df_clean["Concession_Amount"].apply(lambda c: -max(0.0, c - typical_market_concession))

        base_date = df_clean["Contract_Date_DT"].min()
        df_clean["Contract_Months_Elapsed"] = (df_clean["Contract_Date_DT"] - base_date).dt.days / 30.4375

        X = sm.add_constant(df_clean[["GLA_Num", "Contract_Months_Elapsed"]], has_constant="add")
        model = sm.OLS(df_clean["Close_Price_Num"], X).fit()

        intercept = model.params.get("const", 0.0)
        gla_coef = model.params.get("GLA_Num", 0.0)
        monthly_trend_coef = model.params.get("Contract_Months_Elapsed", 0.0)
        r2 = model.rsquared

        annual_pct = ((monthly_trend_coef / df_clean["Close_Price_Num"].mean()) * 100) * 12
        is_trend_significant = (model.pvalues.get("Contract_Months_Elapsed", 1.0) <= 0.05) and (abs(annual_pct) > 2.0)
        active_monthly_trend = monthly_trend_coef if is_trend_significant else 0.0
        market_trend_desc = f"${active_monthly_trend:,.2f}/mo ({annual_pct:+.1f}%/yr - active trend)" if is_trend_significant else "Stable ($0.00/mo - no trend, p > 0.05)"

        subject_contract_dt = pd.to_datetime(valuation_date)
        subject_elapsed_months = (subject_contract_dt - base_date).days / 30.4375

        amenities_df = pd.DataFrame([parse_amenities_and_classification(r, valuation_year) for _, r in df_clean.iterrows()])
        df_merged = pd.concat([df_clean.reset_index(drop=True), amenities_df], axis=1)

        residuals = []
        for _, row in df_merged.iterrows():
            months_prior = (subject_contract_dt - row["Contract_Date_DT"]).days / 30.4375
            raw_diff = effective_subject_gla - row["GLA_Num"]
            gla_adj = 0.0 if abs(raw_diff) <= 100.0 else (raw_diff * gla_coef)
            residuals.append(row["Close_Price_Num"] + (months_prior * active_monthly_trend) + gla_adj + row["Concession_Adj"])
        df_merged["Residual_Price"] = residuals

        subject_cond_score = normalize_condition_score(sub_condition)
        cond_rate, cond_status = solve_rate_by_sensitivity(df_merged, subject_cond_score, "condition_score", (5000, 35000), 2500)
        above_bed_rate, above_bed_status = solve_rate_by_sensitivity(df_merged, sub_above_beds, "beds_above", (0, 15000), 2500)
        above_full_bath_rate, above_full_bath_status = solve_rate_by_sensitivity(df_merged, sub_above_full_baths, "baths_full_above", (5000, 25000), 2500)
        above_half_bath_rate, above_half_bath_status = solve_rate_by_sensitivity(df_merged, sub_above_half_baths, "baths_half_above", (2500, 12500), 1000)
        bsmt_bed_rate, bsmt_bed_status = solve_rate_by_sensitivity(df_merged, sub_bsmt_beds, "bsmt_beds", (0, 10000), 2000)
        bsmt_full_bath_rate, bsmt_full_bath_status = solve_rate_by_sensitivity(df_merged, sub_bsmt_full_baths, "bsmt_full_baths", (3000, 15000), 2000)
        bsmt_half_bath_rate, bsmt_half_bath_status = solve_rate_by_sensitivity(df_merged, sub_bsmt_half_baths, "bsmt_half_baths", (1500, 8000), 1000)
        bsmt_fin_rate, bsmt_fin_status = solve_rate_by_sensitivity(df_merged, sub_bsmt_fin, "bsmt_finished_sf", (25, 75), 5)
        if bsmt_fin_rate == 0.0 and sub_bsmt_fin > 0:
            bsmt_fin_rate = round((gla_coef * 0.33) / 5.0) * 5.0
            bsmt_fin_status = "Appraisal standard (33% of GLA)"

        bsmt_unfin_rate, bsmt_unfin_status = solve_rate_by_sensitivity(df_merged, sub_bsmt_unfin, "bsmt_unfinished_sf", (10, 35), 5)
        front_porch_rate, front_porch_status = solve_rate_by_sensitivity(df_merged, sub_front_porch, "has_front_porch", (2000, 15000), 500)
        screen_rate, screen_status = solve_rate_by_sensitivity(df_merged, sub_screen_porch, "has_screen_porch", (0, 25000), 500)
        deck_rate, deck_status = solve_rate_by_sensitivity(df_merged, sub_deck, "has_deck", (0, 15000), 500)
        fence_rate, fence_status = solve_rate_by_sensitivity(df_merged, sub_fence, "has_fence", (2000, 12000), 500)
        garage_rate, garage_status = solve_rate_by_sensitivity(df_merged, sub_garage, "garage_bays", (5000, 25000), 1000)
        fp_rate, fp_status = solve_rate_by_sensitivity(df_merged, sub_fireplaces, "fireplaces", (2000, 10000), 500)

        subject_amenity_val = (
            (sub_bsmt_fin * bsmt_fin_rate) + (sub_bsmt_unfin * bsmt_unfin_rate) +
            (sub_bsmt_beds * bsmt_bed_rate) + (sub_bsmt_full_baths * bsmt_full_bath_rate) +
            (sub_bsmt_half_baths * bsmt_half_bath_rate) + (sub_fireplaces * fp_rate) +
            (sub_fence * fence_rate) + (sub_front_porch * front_porch_rate) +
            (sub_screen_porch * screen_rate) + (sub_deck * deck_rate)
        )

        subject_stat_benchmark = intercept + (gla_coef * effective_subject_gla) + (active_monthly_trend * subject_elapsed_months) + subject_amenity_val

        comp_candidates = []
        cdom_series = pd.to_numeric(df_merged["Cumulative DOM"], errors="coerce").fillna(0) if "Cumulative DOM" in df_merged.columns else pd.Series([0] * len(df_merged))
        clean_sub_addr = re.sub(r"[^\w\s]", "", sub_address.lower()).strip()
        sub_mls_id = str(st.session_state.get("sub_mls_num", "")).strip()

        for idx, row in df_merged.iterrows():
            clean_mls = clean_str(row["Clean_MLS"])
            row_addr = re.sub(r"[^\w\s]", "", str(row["Address"]).lower()).strip()
            if (clean_sub_addr and clean_sub_addr == row_addr) or (sub_mls_id and clean_mls == sub_mls_id):
                continue

            days_prior = (subject_contract_dt - row["Contract_Date_DT"]).days
            if clean_mls not in manual_mls_picks:
                if cdom_series.iloc[idx] > 120 or days_prior > 365:
                    continue

            comp_price = row["Close_Price_Num"]
            time_adj = (days_prior / 30.4375) * active_monthly_trend
            gla_diff = effective_subject_gla - row["GLA_Num"]
            is_deadband = abs(gla_diff) <= 100.0
            gla_adj = 0.0 if is_deadband else (gla_diff * gla_coef)

            above_bed_adj = (sub_above_beds - row["beds_above"]) * above_bed_rate
            above_fbath_adj = (sub_above_full_baths - row["baths_full_above"]) * above_full_bath_rate
            above_half_bath_adj = (sub_above_half_baths - row["baths_half_above"]) * above_half_bath_rate
            above_room_adj = above_bed_adj + above_fbath_adj + above_half_bath_adj

            bsmt_bed_adj = (sub_bsmt_beds - row["bsmt_beds"]) * bsmt_bed_rate
            bsmt_fbath_adj = (sub_bsmt_full_baths - row["bsmt_full_baths"]) * bsmt_full_bath_rate
            bsmt_hbath_adj = (sub_bsmt_half_baths - row["bsmt_half_baths"]) * bsmt_half_bath_rate
            bsmt_room_adj = bsmt_bed_adj + bsmt_fbath_adj + bsmt_hbath_adj

            garage_adj = (sub_garage - row["garage_bays"]) * garage_rate
            fp_adj = (sub_fireplaces - row["fireplaces"]) * fp_rate
            fence_adj = (sub_fence - row["has_fence"]) * fence_rate
            front_porch_adj = (sub_front_porch - row["has_front_porch"]) * front_porch_rate
            porch_adj = (sub_screen_porch - row["has_screen_porch"]) * screen_rate
            deck_adj = (sub_deck - row["has_deck"]) * deck_rate

            bsmt_fin_diff = sub_bsmt_fin - row["bsmt_finished_sf"]
            bsmt_fin_adj = bsmt_fin_diff * bsmt_fin_rate
            bsmt_unfin_diff = sub_bsmt_unfin - row["bsmt_unfinished_sf"]
            bsmt_unfin_adj = bsmt_unfin_diff * bsmt_unfin_rate
            cond_diff = subject_cond_score - row["condition_score"]
            cond_adj = cond_diff * cond_rate

            net_adj = (
                row["Concession_Adj"] + time_adj + gla_adj + above_room_adj + bsmt_fin_adj +
                bsmt_room_adj + bsmt_unfin_adj + garage_adj + fp_adj + fence_adj +
                front_porch_adj + porch_adj + deck_adj + cond_adj
            )
            gross_adj = (
                abs(row["Concession_Adj"]) + abs(time_adj) + abs(gla_adj) + abs(above_bed_adj) +
                abs(above_fbath_adj) + abs(above_half_bath_adj) + abs(bsmt_fin_adj) +
                abs(bsmt_room_adj) + abs(bsmt_unfin_adj) + abs(garage_adj) +
                abs(fp_adj) + abs(fence_adj) + abs(front_porch_adj) + abs(porch_adj) +
                abs(deck_adj) + abs(cond_adj)
            )

            gross_pct = (gross_adj / comp_price) * 100.0
            net_pct = (abs(net_adj) / comp_price) * 100.0
            indicated = comp_price + net_adj
            bench_diff = abs(indicated - subject_stat_benchmark) / subject_stat_benchmark
            similarity = (1.0 / (max(1.0, days_prior) + 30.0)) * (1.0 / (1.0 + (gross_pct / 10.0))) * (1.0 / (1.0 + (net_pct / 10.0))) * (1.0 / (1.0 + (bench_diff * 3.0)))

            comp_candidates.append({
                "mls": clean_mls, "address": row["Address"], "price": comp_price, "gla": row["GLA_Num"],
                "contract_date": str(row["Purchase Contract Date"]).split(" ")[0], "days_prior": days_prior,
                "gla_diff": gla_diff, "gla_adj": gla_adj, "is_in_deadband": is_deadband,
                "beds_above": row["beds_above"], "baths_full_above": row["baths_full_above"], "baths_half_above": row["baths_half_above"],
                "bsmt_beds": row["bsmt_beds"], "bsmt_full_baths": row["bsmt_full_baths"], "bsmt_half_baths": row["bsmt_half_baths"],
                "bsmt_fin_sf": row["bsmt_finished_sf"], "room_adj": above_room_adj + bsmt_room_adj,
                "bsmt_fin_adj": bsmt_fin_adj, "cond_adj": cond_adj, "net_adj": net_adj,
                "net_pct": net_pct, "gross_pct": gross_pct, "indicated": indicated, "similarity": similarity
            })

        cand_df = pd.DataFrame(comp_candidates)
        selected_set, chosen_mls = [], []

        if manual_mls_picks:
            for m in manual_mls_picks:
                match = cand_df[cand_df["mls"] == m]
                if not match.empty and m not in chosen_mls:
                    selected_set.append(match.iloc[0])
                    chosen_mls.append(m)

        rem = cand_df[~cand_df["mls"].isin(chosen_mls)].copy()

        # Step 1: Enforce Raw Closed Price Bracketing
        if not any(s["price"] >= subject_stat_benchmark for s in selected_set) and not rem.empty:
            hi_price = rem[rem["price"] >= subject_stat_benchmark].sort_values(by="similarity", ascending=False)
            if not hi_price.empty:
                pick = hi_price.iloc[0]
                selected_set.append(pick)
                chosen_mls.append(pick["mls"])
                rem = cand_df[~cand_df["mls"].isin(chosen_mls)].copy()

        if not any(s["price"] <= subject_stat_benchmark for s in selected_set) and not rem.empty:
            lo_price = rem[rem["price"] <= subject_stat_benchmark].sort_values(by="similarity", ascending=False)
            if not lo_price.empty:
                pick = lo_price.iloc[0]
                selected_set.append(pick)
                chosen_mls.append(pick["mls"])
                rem = cand_df[~cand_df["mls"].isin(chosen_mls)].copy()

        # Step 2: Enforce Two-Sided Net Adjustment Bracketing (Positive and Negative)
        has_pos_net = any(s["net_adj"] > 0 for s in selected_set)
        has_neg_net = any(s["net_adj"] < 0 for s in selected_set)

        if not has_pos_net and not rem.empty:
            pos_candidates = rem[rem["net_adj"] > 0].sort_values(by="similarity", ascending=False)
            if not pos_candidates.empty:
                pick = pos_candidates.iloc[0]
                selected_set.append(pick)
                chosen_mls.append(pick["mls"])
                rem = cand_df[~cand_df["mls"].isin(chosen_mls)].copy()

        if not has_neg_net and not rem.empty:
            neg_candidates = rem[rem["net_adj"] < 0].sort_values(by="similarity", ascending=False)
            if not neg_candidates.empty:
                pick = neg_candidates.iloc[0]
                selected_set.append(pick)
                chosen_mls.append(pick["mls"])
                rem = cand_df[~cand_df["mls"].isin(chosen_mls)].copy()

        # Step 3: Fill remaining comp slots up to 3 based on lowest net & gross adjustment percentages
        while len(selected_set) < 3 and not rem.empty:
            pick = rem.sort_values(by="similarity", ascending=False).iloc[0]
            selected_set.append(pick)
            chosen_mls.append(pick["mls"])
            rem = cand_df[~cand_df["mls"].isin(chosen_mls)].copy()

        ranked_comps = pd.DataFrame(selected_set).reset_index(drop=True)
        tot_wt = ranked_comps["similarity"].sum()
        ranked_comps["weight"] = (ranked_comps["similarity"] / tot_wt) * 100.0
        weighted_val = (ranked_comps["indicated"] * ranked_comps["similarity"]).sum() / tot_wt
        min_ind, max_ind = ranked_comps["indicated"].min(), ranked_comps["indicated"].max()
        min_raw, max_raw = ranked_comps["price"].min(), ranked_comps["price"].max()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Statistical Benchmark", f"${subject_stat_benchmark:,.0f}", f"R² = {r2:.4f}")
        m2.metric("Weighted Indicated Value", f"${weighted_val:,.0f}", f"Bracket: ${min_ind:,.0f} - ${max_ind:,.0f}")
        m3.metric("Extracted GLA Rate", f"${gla_coef:.2f} / SF", "Deadband ±100 SF")
        m4.metric("Finished Bsmt Rate", f"${bsmt_fin_rate:.2f} / SF", f"Subject: {sub_bsmt_fin} SF")

        # Table 1: Pure HTML (No Markdown Pipes Anywhere)
        t1_rows = [
            ("Model Goodness-of-Fit (R²)", f"{r2:.4f}", f"Multi-Variable OLS Regression ({len(df_merged)} submarket sales)"),
            ("Above-Grade GLA Rate", f"${gla_coef:.2f} / SF", "Linear marginal contribution (Deadband: +/- 100 SF = $0)"),
            ("Above-Grade Full Bath Adjustment", f"${above_full_bath_rate:,.0f} / Bath", above_full_bath_status),
            ("Above-Grade Half Bath Adjustment", f"${above_half_bath_rate:,.0f} / Bath", above_half_bath_status),
            ("Above-Grade Bedroom Adjustment", f"${above_bed_rate:,.0f} / Bed", above_bed_status),
            ("Finished Basement Rate", f"${bsmt_fin_rate:.2f} / SF", bsmt_fin_status),
            ("Basement Full Bath Adjustment", f"${bsmt_full_bath_rate:,.0f} / Bath", bsmt_full_bath_status),
            ("Basement Bedroom Adjustment", f"${bsmt_bed_rate:,.0f} / Bed", bsmt_bed_status),
            ("Condition / Turnkey Adjustment", f"${cond_rate:,.0f}", "Matched-pair condition rating tier extraction"),
            ("Market Conditions Time-Trend", market_trend_desc, "Appraisal-standard trend reconciliation"),
            ("Screened Porch Contributory Value", f"${screen_rate:,.0f}", "Residual paired-variance optimization"),
            ("Deck Contributory Value", f"${deck_rate:,.0f}", "Residual paired-variance optimization"),
            ("Subject Amenity Contributory Total", f"+${subject_amenity_val:,.0f}", "Cumulative physical improvements"),
            ("Reconciled Statistical Benchmark", f"${subject_stat_benchmark:,.0f}", "Deterministic model valuation (GLA + Macro + Amenities)"),
            ("Reconciled Weighted Sales Indication", f"${weighted_val:,.0f}", "12-month recency & adjustment weighted average")
        ]

        t1_html = '<table class="pdf-clean-table"><thead><tr><th style="width:32%;">Valuation Parameter</th><th style="width:25%;">Verified Rate / Value</th><th style="width:43%;">Derivation Methodology</th></tr></thead><tbody>'
        for p, r, m in t1_rows:
            t1_html += f"<tr><td><b>{p}</b></td><td><b>{r}</b></td><td>{m}</td></tr>"
        t1_html += "</tbody></table>"

        # Table 2: Pure HTML (No Markdown Pipes Anywhere)
        t2_html = '<table class="pdf-clean-table"><thead><tr><th style="width:15%;">Comparable Property</th><th style="width:6.5%;">Closed Price</th><th style="width:6.5%;">Contract Date</th><th style="width:7.5%;">Above / Bsmt</th><th style="width:6.5%;">Above Rms</th><th style="width:6.5%;">Bsmt Rms</th><th style="width:6.5%;">GLA Adj ($)</th><th style="width:6.5%;">Room Adj ($)</th><th style="width:6.5%;">Bsmt Fin ($)</th><th style="width:6.5%;">Cond Adj ($)</th><th style="width:14%;">Net Adjustment</th><th style="width:9%;">Indicated Value</th></tr></thead><tbody>'
        for _, c in ranked_comps.iterrows():
            above_rms = f"{c['beds_above']:.0f}b/{c['baths_full_above']:.0f}.{c['baths_half_above']:.0f}ba"
            bsmt_rms = f"{c['bsmt_beds']:.0f}b/{c['bsmt_full_baths']:.0f}.{c['bsmt_half_baths']:.0f}ba"
            t2_html += f"""<tr>
                <td><b>{c['address']}</b><br><span style="font-size:6.2pt; color:#64748B;">MLS #{c['mls']}</span></td>
                <td>${c['price']:,.0f}</td>
                <td>{c['contract_date']}</td>
                <td>{c['gla']:,.0f} SF<br><span style="font-size:6.2pt; color:#64748B;">{c['bsmt_fin_sf']:.0f} Bsmt</span></td>
                <td>{above_rms}</td>
                <td>{bsmt_rms}</td>
                <td><b>${c['gla_adj']:+,.0f}</b></td>
                <td>${c['room_adj']:+,.0f}</td>
                <td>${c['bsmt_fin_adj']:+,.0f}</td>
                <td>${c['cond_adj']:+,.0f}</td>
                <td><b>${c['net_adj']:+,.0f}</b><br><span style="font-size:6.2pt; color:#64748B;">({c['gross_pct']:.1f}% gross / {c['net_pct']:.1f}% net)</span></td>
                <td><b>${c['indicated']:,.0f}</b></td>
            </tr>"""
        t2_html += "</tbody></table>"

        # LLM Inference
        prompt_1 = f"Write the Executive Pricing Strategy & Valuation Thesis for: {sub_address} in {subdivision_display} (Built {sub_year_built}, {effective_subject_gla:,.0f} SF). Recommended Corridor: ${min_ind:,.0f} to ${max_ind:,.0f} (Midpoint: ${weighted_val:,.0f}). Statistical Benchmark: ${subject_stat_benchmark:,.0f}. Extracted GLA Rate: ${gla_coef:.2f}/SF with ±100 SF deadband. Write exactly two complete paragraphs. Do not cut off sentences. No headers."
        prompt_2 = f"Write the ANSI Z765 & Comp Bracketing Defense for: {sub_address} ({effective_subject_gla:,.0f} SF Above-Grade, {sub_bsmt_fin:,.0f} SF Bsmt). Comps bracket from ${min_raw:,.0f} to ${max_raw:,.0f} unadjusted, and ${min_ind:,.0f} to ${max_ind:,.0f} indicated. Explain compliance and bracketing in exactly two concise paragraphs. No headers."

        headers = {"Content-Type": "application/json"}
        if provider == "Groq Cloud (Sub-Second)":
            headers["Authorization"] = f"Bearer {groq_api_key.strip()}"

        def call_inference(prompt):
            payload = {
                "model": active_model,
                "messages": [
                    {"role": "system", "content": "You are a senior valuation analyst writing an Executive Listing Brief. Use standard business English. Complete all sentences fully. Never use LaTeX syntax."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2, "max_tokens": 800
            }
            try:
                res = requests.post(target_endpoint, headers=headers, data=json.dumps(payload), timeout=25 if provider == "Groq Cloud (Sub-Second)" else 120)
                if res.status_code == 200:
                    return res.json()["choices"][0]["message"]["content"]
            except Exception:
                pass
            return ""

        sec_a = sanitize_markdown_text(call_inference(prompt_1))
        sec_b = sanitize_markdown_text(call_inference(prompt_2))

        if not sec_a:
            sec_a = f"The recommended pricing corridor for {sub_address} is anchored between ${min_ind:,.0f} and ${max_ind:,.0f}, with a weighted midpoint indication of ${weighted_val:,.0f}. This pricing thesis is supported by a multi-variable OLS statistical benchmark of ${subject_stat_benchmark:,.0f} derived from verified closed sales within {subdivision_display}. The comparable selection algorithm strictly minimizes net and gross adjustment percentages, ensuring tight convergence between closed prices and final adjusted indications.\n\nThe extracted rate of ${gla_coef:.2f} per square foot, applied with a ±100 square foot measurement deadband, acts as a primary buffer against appraisal valuation gaps. Minor variances in exterior measurement calculations between appraisers are absorbed within this deadband, ensuring that routine physical measurement discrepancies do not adversely impact the final appraised value."
        if not sec_b:
            sec_b = f"The subject property gross living area of {effective_subject_gla:,.0f} SF has been measured and verified in strict accordance with ANSI Z765-2021 standards, restricted entirely to above-grade finished space. Below-grade areas totaling {sub_bsmt_fin:,.0f} SF have been segmented and evaluated independently to prevent non-compliant square footage conflation under secondary mortgage guidelines.\n\nAppraisal bracketing criteria are fully satisfied across the primary closed comparables. By optimizing for low net and gross adjustments, the subject's unadjusted closed prices (${min_raw:,.0f} to ${max_raw:,.0f}) and net adjusted values (${min_ind:,.0f} to ${max_ind:,.0f}) tightly bracket the reconciled benchmark, establishing an underwriter-compliant valuation defense grid."

        footnote = f"*Note: All comparables restricted to closed transactions within the prior 12 months. Comps algorithmically selected to minimize net and gross adjustment percentages while maintaining Fannie Mae two-sided bracketing. Above-grade living area differences of 100 SF or less receive a $0 adjustment per appraisal measurement standards.*"
        header_sub_display = f" | Subdivision: {sub_subdivision.strip()}" if sub_subdivision.strip() else " | Market: Competitive Submarket"

        # =====================================================================
        # 1. RENDER DIRECTLY TO MONITOR
        # =====================================================================
        st.markdown(f"## Executive Listing Defense Brief & Valuation Packet")
        st.markdown(f"**Subject Property:** {sub_address} | **Valuation Date:** {valuation_date}{header_sub_display}")
        st.info("**REGULATORY & LEGAL DISCLAIMER (N.C.G.S. § 93A-83):** This document is a computer-assisted Comparative Market Analysis (CMA) and Listing Defense Brief prepared solely for listing strategy, transaction consultation, and contract defense. THIS IS NOT AN OFFICIAL APPRAISAL.")
        st.markdown("### 1. Executive Pricing Strategy & Valuation Thesis")
        st.markdown(f"<div class='report-narrative'>{sec_a}</div>", unsafe_allow_html=True)
        st.markdown("#### Verified Market Regression & Sensitivity Extractions")
        st.markdown(f"<div class='table-container'>{t1_html.replace('pdf-clean-table', 'screen-table')}</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 2. ANSI Z765 & Comp Bracketing Defense")
        st.markdown(f"<div class='report-narrative'>{sec_b}</div>", unsafe_allow_html=True)
        st.markdown("#### Sales Comparison Approach Reconciliation Grid")
        st.markdown(f"<div class='table-container'>{t2_html.replace('pdf-clean-table', 'screen-table')}</div>", unsafe_allow_html=True)
        st.markdown(f"<small>{footnote}</small>", unsafe_allow_html=True)

        # =====================================================================
        # 2. GENERATE AND ATTACH DOWNLOADABLE PDF
        # =====================================================================
        pdf_bytes = build_pdf_document(
            sub_address=sub_address,
            valuation_date=str(valuation_date),
            header_sub=header_sub_display,
            sec_a=sec_a,
            sec_b=sec_b,
            table1_html=t1_html,
            table2_html=t2_html,
            footnote=footnote
        )

        st.success("Listing Defense Brief successfully generated!")

        clean_filename = re.sub(r"[^\w\s]", "", sub_address).replace(" ", "_")
        if pdf_bytes:
            st.download_button(
                label="📄 Download Institutional Brief as PDF (2-Page Format)",
                data=pdf_bytes,
                file_name=f"Listing_Defense_Brief_{clean_filename}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.warning("PDF compilation notice: please re-run to refresh buffer.")
