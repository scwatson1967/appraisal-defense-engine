import os
import re
import math
from io import BytesIO
from datetime import date
import requests
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from xhtml2pdf import pisa
from groq import Groq

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="8-County NC Valuation & Listing Defense Engine",
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
# 8-COUNTY NC ADAPTIVE RADIUS CONFIGURATION
# ==============================================================================
NC_COUNTY_CONFIG = {
    "Wake": {"default_radius": 0.75, "tier": "Urban/Suburban Core"},
    "Durham": {"default_radius": 0.75, "tier": "Urban/Suburban Core"},
    "Cumberland": {"default_radius": 1.00, "tier": "Metro Military Core"},
    "Orange": {"default_radius": 1.25, "tier": "Mixed Suburban/Rural"},
    "Johnston": {"default_radius": 1.50, "tier": "Suburban/Exurban Growth"},
    "Chatham": {"default_radius": 2.25, "tier": "Suburban/Rural Acreage"},
    "Franklin": {"default_radius": 2.50, "tier": "Rural/Exurban"},
    "Granville": {"default_radius": 2.50, "tier": "Rural/Exurban"},
}

# ==============================================================================
# TEXT SANITIZER
# ==============================================================================
def sanitize_for_pdf(text: str) -> str:
    if not isinstance(text, str):
        return str(text)

    replacements = {
        r"\pm": "+/-", "±": "+/-", r"\approx": "~", "≈": "~",
        r"\times": "x", "×": "x", r"\le": "<=", r"\ge": ">=",
        "–": "-", "—": "-", "“": '"', "”": '"', "’": "'", "‘": "'",
        "…": "...", "•": "|", "\u00a0": " ", "\u200b": "",
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
def clean_currency(val) -> float:
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    val_str = re.sub(r'[^\d.]', '', str(val))
    return float(val_str) if val_str else 0.0

def map_prop_type(val) -> str:
    """Aggressively standardizes messy MLS and Tax records into 3 core appraisal buckets."""
    v = str(val).lower().strip()
    if not v or v in ['nan', 'none']: return 'Detached'
    if any(x in v for x in ['condo', 'condominium', 'highrise', 'apartment']): return 'Condo'
    if any(x in v for x in ['town', 'attached', 'duplex', 'row', 'twin', 'half']): return 'Attached'
    # Defaults anything like "Single Family", "SFR", "RES", "Residential" to Detached
    return 'Detached'

def load_and_clean_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df['MLS #'] = df['MLS #'].astype(str).str.strip() if 'MLS #' in df.columns else [f"MLS-{i+1}" for i in range(len(df))]

    addr_cols = ['Street Address', 'Full Address', 'Property Address', 'Address']
    found_addr = next((c for c in addr_cols if c in df.columns), None)
    df['Address'] = df[found_addr].astype(str) if found_addr else "Unknown Address"

    price_cols = ['Close Price', 'Sold Price', 'Purchase Price', 'Price']
    found_price = next((c for c in price_cols if c in df.columns), None)
    df['Clean_Price'] = df[found_price].apply(clean_currency) if found_price else 0.0
    df = df[df['Clean_Price'] > 20000].copy()

    gla_cols = ['Living Area', 'Above Grade Finished Area', 'Gross Living Area', 'SqFt']
    found_gla = next((c for c in gla_cols if c in df.columns), None)
    df['Clean_GLA'] = df[found_gla].apply(clean_currency) if found_gla else 0.0

    bsmt_cols = ['Below Grade Finished Area', 'Basement Finished SqFt', 'Finished Basement']
    found_bsmt = next((c for c in bsmt_cols if c in df.columns), None)
    df['Clean_Bsmt_Fin'] = df[found_bsmt].apply(clean_currency) if found_bsmt else 0.0

    type_cols = ['Property Sub Type', 'Property Type', 'Type', 'Sub Type']
    found_type = next((c for c in type_cols if c in df.columns), None)
    df['Clean_Property_Type'] = df[found_type].apply(map_prop_type) if found_type else 'Detached'

    yr_cols = ['Year Built', 'Yr Built', 'Built']
    found_yr = next((c for c in yr_cols if c in df.columns), None)
    df['Clean_Year'] = pd.to_numeric(df[found_yr], errors='coerce').fillna(2000).astype(int) if found_yr else 2000

    bed_cols = ['Bedrooms Total', 'Bedrooms', 'Beds Total']
    found_bed = next((c for c in bed_cols if c in df.columns), None)
    df['Clean_Beds'] = pd.to_numeric(df[found_bed], errors='coerce').fillna(3).astype(int) if found_bed else 3

    fb_cols = ['Bathrooms Full', 'Baths Full', 'Full Baths']
    found_fb = next((c for c in fb_cols if c in df.columns), None)
    df['Clean_FB'] = pd.to_numeric(df[found_fb], errors='coerce').fillna(2).astype(int) if found_fb else 2

    hb_cols = ['Bathrooms Half', 'Baths Half', 'Half Baths']
    found_hb = next((c for c in hb_cols if c in df.columns), None)
    df['Clean_HB'] = pd.to_numeric(df[found_hb], errors='coerce').fillna(0).astype(int) if found_hb else 0

    subd_cols = ['Subdivision', 'Subdivision-Free Text', 'Legal Subdivision']
    found_subd = next((c for c in subd_cols if c in df.columns), None)
    df['Clean_Subdivision'] = df[found_subd].fillna("").astype(str).str.strip() if found_subd else ""

    date_cols = ['Close Date', 'Closed Date', 'Sold Date']
    found_date = next((c for c in date_cols if c in df.columns), None)
    df['Clean_Date'] = pd.to_datetime(df[found_date], errors='coerce') if found_date else pd.to_datetime('today')

    garage_cols = ['Garage Spaces', 'Garage Total', 'Garage']
    found_gar = next((c for c in garage_cols if c in df.columns), None)
    df['Clean_Garage'] = df[found_gar].apply(clean_currency) if found_gar else 0.0

    return df

def fetch_comps_from_nconemap(address: str, radius_miles: float) -> tuple[dict, pd.DataFrame]:
    geo_url = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
    geo_params = {"SingleLine": address.strip(), "f": "json", "outFields": "Match_addr,Subregion,City,Postal", "maxLocations": 1}
    g_resp = requests.get(geo_url, params=geo_params, timeout=15)
    g_resp.raise_for_status()
    candidates = g_resp.json().get("candidates", [])
    if not candidates:
        raise ValueError("Address could not be geocoded. Please include Street, City, NC, and ZIP.")

    best = candidates[0]
    lon = float(best["location"]["x"])
    lat = float(best["location"]["y"])
    attrs = best.get("attributes", {})
    detected_county = str(attrs.get("Subregion") or "Wake").replace(" County", "").strip()
    if detected_county not in NC_COUNTY_CONFIG: detected_county = "Wake"

    lat_deg = float(radius_miles) / 69.0
    lon_deg = float(radius_miles) / (69.0 * max(math.cos(math.radians(lat)), 0.1))
    envelope = f"{lon - lon_deg},{lat - lat_deg},{lon + lon_deg},{lat + lat_deg}"

    subject_autofill = {
        "address": attrs.get("Match_addr") or address.strip(),
        "county": detected_county,
        "subdivision": "",
        "property_type": "Detached",
        "year_built": 2000, "gla": 0, "beds": 3, "baths_full": 2, "baths_half": 0,
    }

    street_num = address.strip().split(" ")[0].lower()
    rows = []

    if detected_county == "Wake":
        try:
            wake_url = "https://services1.arcgis.com/a7CWfuGP5ZnLYE7I/arcgis/rest/services/Wake_Parcels_Public/FeatureServer/0/query"
            w_params = {
                "geometry": envelope, "geometryType": "esriGeometryEnvelope", "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects", "where": "HEATEDAREA > 500",
                "outFields": "PIN_NUM,SITE_ADDRESS,HEATEDAREA,YEAR_BUILT,TOTSALPRICE,TOTAL_VALUE_ASSD,SALE_DATE,MAP_NAME,TYPE_USE_DECODE",
                "returnGeometry": "false", "resultRecordCount": 150, "f": "json"
            }
            w_resp = requests.get(wake_url, params=w_params, timeout=15)
            if w_resp.status_code == 200 and "features" in w_resp.json():
                for idx, feat in enumerate(w_resp.json().get("features", [])):
                    a = feat.get("attributes") or {}
                    site_addr = str(a.get("SITE_ADDRESS") or f"Wake Parcel {idx+1}").strip()
                    heated = float(a.get("HEATEDAREA") or 0)
                    yr_built = int(a.get("YEAR_BUILT") or 2000)
                    subd = str(a.get("MAP_NAME") or "").strip()
                    ptype = map_prop_type(a.get("TYPE_USE_DECODE", ""))

                    sale_price = float(a.get("TOTSALPRICE") or 0)
                    if sale_price <= 35000: sale_price = float(a.get("TOTAL_VALUE_ASSD") or 0)
                    if sale_price <= 35000 or heated <= 500: continue

                    raw_dt = a.get("SALE_DATE")
                    clean_dt = pd.to_datetime(raw_dt, unit="ms", errors="coerce") if isinstance(raw_dt, (int, float)) and raw_dt > 0 else pd.to_datetime(raw_dt, errors="coerce")
                    if pd.isna(clean_dt): clean_dt = pd.to_datetime("today") - pd.Timedelta(days=120)

                    if site_addr.lower().startswith(street_num + " "):
                        subject_autofill["gla"] = int(heated)
                        subject_autofill["property_type"] = ptype
                        if 1800 < yr_built <= 2027: subject_autofill["year_built"] = yr_built
                        if subd: subject_autofill["subdivision"] = subd
                        continue

                    rows.append({
                        "MLS #": str(a.get("PIN_NUM") or f"WAKE-{idx+1}"),
                        "Address": site_addr, "Clean_Price": sale_price, "Clean_GLA": heated, "Clean_Bsmt_Fin": 0.0,
                        "Clean_Year": yr_built if 1800 < yr_built <= 2027 else 2000,
                        "Clean_Beds": 3, "Clean_FB": 2, "Clean_HB": 1, "Clean_Subdivision": subd,
                        "Clean_Date": clean_dt, "Clean_Garage": 2.0, "Clean_Property_Type": ptype,
                    })
        except Exception:
            pass

    if not rows:
        parcel_url = "https://services.nconemap.gov/secure/rest/services/NC1Map_Parcels/MapServer/0/query"
        p_params = {
            "geometry": envelope, "geometryType": "esriGeometryEnvelope", "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects", "where": "parval > 40000 AND improvval > 20000",
            "outFields": "parno,siteadd,scity,parval,improvval,landval,saledate,gisacres,cntyname,parusedesc",
            "returnGeometry": "false", "resultRecordCount": 150, "f": "json"
        }
        p_resp = requests.get(parcel_url, params=p_params, timeout=20)
        p_resp.raise_for_status()
        
        for idx, feat in enumerate(p_resp.json().get("features", [])):
            a = feat.get("attributes") or {}
            par_val = float(a.get("parval") or a.get("PARVAL") or 0)
            imp_val = float(a.get("improvval") or a.get("IMPROVVAL") or 0)
            if par_val <= 40000 or imp_val <= 20000: continue

            site_addr = str(a.get("siteadd") or a.get("SITEADD") or f"Parcel {a.get('parno', idx+1)}").strip()
            est_gla = round(max(imp_val / 165.0, 900.0), -1)
            ptype = map_prop_type(a.get("parusedesc", ""))

            raw_dt = a.get("saledate") or a.get("SALEDATE")
            clean_dt = pd.to_datetime(raw_dt, unit="ms", errors="coerce") if isinstance(raw_dt, (int, float)) and raw_dt > 0 else pd.to_datetime(raw_dt, errors="coerce")
            if pd.isna(clean_dt): clean_dt = pd.to_datetime("today") - pd.Timedelta(days=120)

            if site_addr.lower().startswith(street_num + " "):
                if subject_autofill["gla"] == 0: subject_autofill["gla"] = int(est_gla)
                subject_autofill["property_type"] = ptype
                continue

            rows.append({
                "MLS #": str(a.get("parno") or a.get("PARNO") or f"NCMAP-{idx+1}"),
                "Address": site_addr, "Clean_Price": par_val, "Clean_GLA": est_gla, "Clean_Bsmt_Fin": 0.0,
                "Clean_Year": 2000, "Clean_Beds": 3, "Clean_FB": 2, "Clean_HB": 0, "Clean_Subdivision": "",
                "Clean_Date": clean_dt, "Clean_Garage": 2.0, "Clean_Property_Type": ptype,
            })

    return subject_autofill, pd.DataFrame(rows)

# ==============================================================================
# EMPIRICAL MARKET CONDITIONS & REGRESSION ENGINE
# ==============================================================================
def calculate_historical_market_trend(df: pd.DataFrame, subject: dict) -> dict:
    val_dt = pd.to_datetime(subject['val_date'])
    trend_df = df[(df['Clean_GLA'] > 500) & (df['Clean_Price'] > 20000) & (df['Clean_Date'].notnull()) & (df['Clean_Property_Type'] == subject['property_type'])].copy()

    trend_df['Months_Ago'] = (val_dt - trend_df['Clean_Date']).dt.days / 30.4375
    recent_df = trend_df[(trend_df['Months_Ago'] >= 0) & (trend_df['Months_Ago'] <= 18)].copy()

    if len(recent_df) < 5:
        return {"monthly_pct": 0.0, "annual_pct": 0.0, "sample_size": len(recent_df), "span_days": 0, "status": f"Insufficient {subject['property_type']} sales (<5) for trend; default 0.00%/mo."}

    span_days = int((recent_df['Clean_Date'].max() - recent_df['Clean_Date'].min()).days)
    if span_days < 45:
        return {"monthly_pct": 0.0, "annual_pct": 0.0, "sample_size": len(recent_df), "span_days": span_days, "status": f"Date span too narrow ({span_days} days); default 0.00%/mo."}

    recent_df['PPSF'] = recent_df['Clean_Price'] / recent_df['Clean_GLA']
    q_low, q_high = recent_df['PPSF'].quantile([0.05, 0.95])
    trimmed = recent_df[(recent_df['PPSF'] >= q_low) & (recent_df['PPSF'] <= q_high)].copy()
    if len(trimmed) < 5: trimmed = recent_df

    try:
        X = sm.add_constant(trimmed['Months_Ago'], has_constant='add')
        y = trimmed['PPSF']
        model = sm.OLS(y, X).fit()
        slope_per_month_ago = float(model.params.get('Months_Ago', 0.0))
        median_ppsf = float(trimmed['PPSF'].median())

        raw_monthly_pct = (-slope_per_month_ago / median_ppsf) * 100.0 if median_ppsf > 0 else 0.0
        monthly_pct = round(float(np.clip(raw_monthly_pct, -0.75, 0.75)), 2)
        direction = "declining" if monthly_pct < -0.04 else ("appreciating" if monthly_pct > 0.04 else "stable")
        return {"monthly_pct": monthly_pct, "annual_pct": round(monthly_pct * 12.0, 1), "sample_size": len(trimmed), "span_days": span_days, "status": f"{subject['property_type']} trend ({len(trimmed)} sales): {direction} at {monthly_pct:+.2f}%/mo."}
    except Exception:
        return {"monthly_pct": 0.0, "annual_pct": 0.0, "sample_size": len(recent_df), "span_days": span_days, "status": "Trend regression fallback: 0.00%/mo."}

def run_regression_analysis(df: pd.DataFrame, subject: dict) -> dict:
    reg_df = df[(df['Clean_GLA'] > 500) & (df['Clean_Property_Type'] == subject['property_type'])].copy()
    if len(reg_df) < 5:
        median_ppsf = (reg_df['Clean_Price'] / reg_df['Clean_GLA']).median() if not reg_df.empty else 185.0
        return {"r2": 0.50, "gla_rate": round(float(np.clip(median_ppsf * 0.40, 45.0, 165.0)), 2), "fb_adj": 7500.0, "hb_adj": 3500.0, "bed_adj": 0.0, "garage_rate": 10000.0, "benchmark": round(float(subject['gla'] * median_ppsf), 0)}

    X = sm.add_constant(reg_df[['Clean_GLA', 'Clean_FB', 'Clean_HB']], has_constant='add')
    try:
        model = sm.OLS(reg_df['Clean_Price'], X).fit()
        gla_rate = round(float(np.clip(model.params.get('Clean_GLA', 55.0), 35.0, 185.0)), 2)
        fb_adj = round(float(np.clip(model.params.get('Clean_FB', 6000.0), 3000.0, 25000.0)), 0)
        hb_adj = round(float(np.clip(model.params.get('Clean_HB', 3000.0), 1500.0, 12500.0)), 0)
        benchmark = round(float(model.predict([1.0, subject['gla'], subject['baths_full'], subject['baths_half']])[0]), 0)
    except Exception:
        gla_rate, fb_adj, hb_adj, benchmark = 55.0, 5000.0, 2500.0, round(float(reg_df['Clean_Price'].median()), 0)

    return {"r2": 0.5, "gla_rate": gla_rate, "fb_adj": fb_adj, "hb_adj": hb_adj, "bed_adj": 0.0, "garage_rate": 10000.0, "benchmark": benchmark}

# ==============================================================================
# COMPARABLE SELECTION (SMART ELASTIC TIME LIMITS)
# ==============================================================================
def select_comparables_with_subdivision(df: pd.DataFrame, subject: dict, reg_params: dict, mode: str = "algorithmic", manual_mls_ids=None) -> pd.DataFrame:
    df_pool = df.copy()
    val_dt = pd.to_datetime(subject['val_date'])
    
    # 1. STRICT PROPERTY TYPE MATCH (No mixing condos with detached)
    df_pool = df_pool[df_pool['Clean_Property_Type'] == subject['property_type']].copy()
    if df_pool.empty: return df_pool

    df_pool['days_old'] = (val_dt - df_pool['Clean_Date']).dt.days

    if mode == "manual" and manual_mls_ids:
        return df_pool[df_pool['MLS #'].astype(str).isin([str(x) for x in manual_mls_ids])].copy()

    # 2. SMART ELASTIC TIME HORIZON (Prefers <12 months, auto-expands to 24 months if low volume)
    recent_comps = df_pool[(df_pool['days_old'] >= 0) & (df_pool['days_old'] <= 365)]
    if len(recent_comps) >= 4:
        df_pool = recent_comps.copy()
    else:
        # Fall back to 24 months (730 days) to prevent the grid from crashing due to zero comps
        df_pool = df_pool[(df_pool['days_old'] >= 0) & (df_pool['days_old'] <= 730)].copy()

    if df_pool.empty: return df_pool

    subj_gla, subj_subd, benchmark = subject['gla'], str(subject.get('subdivision', '')).strip().lower(), reg_params['benchmark']

    scores = []
    for _, row in df_pool.iterrows():
        gla_diff = abs(row['Clean_GLA'] - subj_gla)
        row_subd = str(row['Clean_Subdivision']).strip().lower()
        subd_bonus = -500.0 if (subj_subd and row_subd == subj_subd) else (-300.0 if (subj_subd and re.search(rf'\b{re.escape(subj_subd)}\b', row_subd)) else 0.0)

        # 3. AGE PENALTY: Prioritizes recent sales, applies steep penalty to anything >12 months
        days = row['days_old']
        if days <= 180: recency_penalty = days * 0.05
        elif days <= 365: recency_penalty = (180 * 0.05) + ((days - 180) * 0.25)
        else: recency_penalty = (180 * 0.05) + (185 * 0.25) + ((days - 365) * 1.0) # Massive penalty for >1yr

        scores.append(gla_diff + subd_bonus + recency_penalty)

    df_pool['score'] = scores
    df_pool = df_pool.sort_values('score')

    selected_mls = []
    if subj_subd:
        for _, r in df_pool[df_pool['Clean_Subdivision'].str.lower().str.strip() == subj_subd].head(2).iterrows(): selected_mls.append(r['MLS #'])

    pool_lower, pool_upper = df_pool[df_pool['Clean_Price'] <= benchmark], df_pool[df_pool['Clean_Price'] >= benchmark]
    if not pool_lower.empty:
        lower_comp = pool_lower[~pool_lower['MLS #'].isin(selected_mls)].head(1)
        if not lower_comp.empty: selected_mls.append(lower_comp.iloc[0]['MLS #'])
    if not pool_upper.empty:
        upper_comp = pool_upper[~pool_upper['MLS #'].isin(selected_mls)].head(1)
        if not upper_comp.empty: selected_mls.append(upper_comp.iloc[0]['MLS #'])

    for _, r in df_pool[~df_pool['MLS #'].isin(selected_mls)].iterrows():
        if len(selected_mls) >= 4: break
        selected_mls.append(r['MLS #'])

    return df_pool[df_pool['MLS #'].isin(selected_mls)].copy()

# ==============================================================================
# SALES COMPARISON GRID ADJUSTMENTS
# ==============================================================================
def calculate_grid_adjustments(comps_df: pd.DataFrame, subject: dict, reg_params: dict) -> pd.DataFrame:
    grid_rows = []
    val_dt, monthly_trend = pd.to_datetime(subject['val_date']), subject.get('monthly_trend_pct', 0.0) / 100.0
    cond_adj_val = {"Excellent": 15000.0, "Good": 7500.0, "Average": 0.0, "Fair": -12500.0}.get(subject.get('condition', 'Average'), 0.0)
    am = subject.get('amenities', {})
    amenity_adj_val = (3500.0 if am.get("Screen Porch") else 0.0) + (2500.0 if am.get("Fenced Yard") else 0.0) + (12000.0 if am.get("Pool") else 0.0) + (max(subject.get('fireplaces', 1) - 1, 0) * 2000.0)

    for _, row in comps_df.iterrows():
        price, comp_gla = row['Clean_Price'], row['Clean_GLA']
        time_adj = round(price * monthly_trend * max((val_dt - row['Clean_Date']).days / 30.4375, 0.0), 0) if pd.notnull(row['Clean_Date']) and monthly_trend != 0.0 else 0.0

        gla_diff = subject['gla'] - comp_gla
        gla_adj = 0.0 if abs(gla_diff) <= 100.0 else round(gla_diff * reg_params['gla_rate'], 0)

        fb_diff, hb_diff = subject['baths_full'] - row['Clean_FB'], subject['baths_half'] - row['Clean_HB']
        if (fb_diff <= 0 and hb_diff >= 0) or (fb_diff < 0): room_adj, room_logic_note = 0.0, "GLA adjustment absorbed functional utility."
        elif fb_diff > 0: room_adj, room_logic_note = round((fb_diff * reg_params.get('fb_adj', 7500.0)) + (hb_diff * reg_params.get('hb_adj', 3500.0)), 0), f"Comp inferior by {fb_diff} bath(s); adjusted up."
        else: room_adj, room_logic_note = 0.0, "Equivalent functional utility."

        gar_adj = round((subject['garage_bays'] - row['Clean_Garage']) * 10000.0, 0)
        bsmt_adj = round((subject['bsmt_fin'] - row.get('Clean_Bsmt_Fin', 0.0)) * 50.0, 0)

        yr_diff = subject['year_built'] - row.get('Clean_Year', subject['year_built'])
        age_adj = round(np.clip(yr_diff * 350.0, -12000.0, 12000.0), 0) if abs(yr_diff) >= 5 else 0.0
        other_adj = round(cond_adj_val + amenity_adj_val + age_adj, 0)

        net_adj = time_adj + gla_adj + room_adj + gar_adj + bsmt_adj + other_adj
        gross_adj = abs(time_adj) + abs(gla_adj) + abs(room_adj) + abs(gar_adj) + abs(bsmt_adj) + abs(other_adj)
        indicated = price + net_adj

        grid_rows.append({
            "mls": row.get('MLS #', 'N/A'), "address": row.get('Address', 'Unknown Address'), "subdivision": row.get('Clean_Subdivision', ''),
            "price": price, "contract_date": row['Clean_Date'].strftime('%m/%d/%Y') if pd.notnull(row['Clean_Date']) else 'N/A',
            "days_old": row.get('days_old', 0),
            "gla": comp_gla, "rooms": f"{row['Clean_Beds']}b/{row['Clean_FB']}.{row['Clean_HB']}ba",
            "time_adj": time_adj, "gla_adj": gla_adj, "room_adj": room_adj, "gar_adj": gar_adj, "bsmt_adj": bsmt_adj, "other_adj": other_adj,
            "net_adj": net_adj, "gross_adj": gross_adj, "net_pct": round((net_adj / price) * 100, 1) if price > 0 else 0.0,
            "gross_pct": round((gross_adj / price) * 100, 1) if price > 0 else 0.0, "indicated": indicated, "room_logic_note": room_logic_note
        })
    return pd.DataFrame(grid_rows)

# ==============================================================================
# GROQ NARRATIVE GENERATOR
# ==============================================================================
def generate_narrative_groq(api_key: str, subject: dict, reg_params: dict, grid_df: pd.DataFrame, trend_info: dict) -> str:
    min_ind, max_ind = int(grid_df['indicated'].min()), int(grid_df['indicated'].max())
    midpoint, benchmark = int((min_ind + max_ind) / 2), int(reg_params['benchmark'])
    
    max_days_old = int(grid_df['days_old'].max()) if 'days_old' in grid_df.columns else 0
    time_caveat = " (Note: Comparable search expanded up to 24 months due to limited recent neighborhood transaction volume, with appropriate historical market condition time adjustments applied)" if max_days_old > 365 else " (All comparables utilized closed within the prior 12 months)"

    subj_bath_str = f"{subject['baths_full']}.{subject['baths_half']}ba"
    subd_label = subject['subdivision'] if subject['subdivision'] else f"{subject['county']} County submarket"

    if subject.get('val_mode', 'Pre-Listing / CMA Strategy') == "Executed Contract Defense":
        thesis_sentence = f"For executed contract review on {subject['address']} in {subd_label}, the sales comparison grid establishes an adjusted value range of ${min_ind:,.0f} to ${max_ind:,.0f} (midpoint ${midpoint:,.0f}) against a statistical hedonic benchmark of ${benchmark:,.0f}, reflecting an empirically derived market trend of {trend_info['monthly_pct']:+.2f}%/month ({trend_info['annual_pct']:+.1f}%/year) across {trend_info['sample_size']} historical {subject['property_type']} transactions. Contracts ratified near or below the corridor midpoint are well-supported by closed neighborhood comparables{time_caveat}, whereas contract prices approaching ${max_ind:,.0f} require strong documentary support for condition, upgrades, and net seller concessions."
    else:
        thesis_sentence = f"The pre-listing market analysis for {subject['address']} in {subd_label} establishes an indicated price range of ${min_ind:,.0f} to ${max_ind:,.0f}, yielding a supported corridor midpoint of ${midpoint:,.0f} against a statistical hedonic benchmark of ${benchmark:,.0f}. Historical sales regression across {trend_info['sample_size']} {subject['property_type']} transactions indicates a market velocity of {trend_info['monthly_pct']:+.2f}%/month ({trend_info['annual_pct']:+.1f}%/year). Positioning list price near the midpoint aligns with current {subject['county']} County buyer absorption{time_caveat}, whereas pricing toward the upper limit (${max_ind:,.0f}) increases days-on-market and lender appraisal-gap risk."

    fallback_narrative = f"<p><b>Executive Pricing Strategy & Market Range Thesis:</b> {thesis_sentence}</p><p><b>ANSI Z765 & Functional Utility Bracketing Analysis:</b> Two-sided physical and price bracketing is satisfied. Because buyers do not pay a premium for fewer baths, and marginal room utility is captured within the ${reg_params['gla_rate']:,.2f}/SF gross living area adjustment, separate room adjustments are reconciled at $0 where appropriate to prevent double-counting.</p>"

    clean_key = str(api_key).strip().strip('"').strip("'")
    if not clean_key or not clean_key.startswith("gsk_"): return fallback_narrative

    try:
        client = Groq(api_key=clean_key)
        active_ids = [m.id for m in client.models.list().data]
        chosen_model = next((m for m in ["openai/gpt-oss-120b", "qwen/qwen3.6-27b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"] if m in active_ids), "openai/gpt-oss-120b")
        comp_lines = "\n".join([f"- {r['address']}: {int(r['gla']):,} SF, Closed ${r['price']:,.0f} ({r.get('days_old', 0)} days ago), Indicated ${r['indicated']:,.0f}" for _, r in grid_df.iterrows()])

        prompt = f"""
DELIVERABLE MODE: {subject.get('val_mode')} | SUBJECT: {subject['address']} | Type: {subject['property_type']} | GLA: {subject['gla']:,} SF
Hedonic Benchmark: ${benchmark:,.0f} | Range: ${min_ind:,.0f} to ${max_ind:,.0f} | Trend: {trend_info['monthly_pct']:+.2f}%/mo
COMPARABLES GRID:
{comp_lines}

Write an executive market pricing brief strictly as TWO HTML paragraphs (<p><b>Title:</b> Text</p>):
1. <p><b>Executive Pricing Strategy:</b> Reconcile the indicated range, midpoint, and benchmark. Mention the {subject['property_type']} market trend. Explicitly include this text exactly if applicable: "{time_caveat}".</p>
2. <p><b>ANSI Z765 & Functional Utility:</b> Explain how the ${reg_params['gla_rate']}/SF adjustment captures living area utility without double-counting bedroom/bath differences.</p>
Do NOT claim this is a USPAP appraisal. Max 250 words.
"""
        response = client.chat.completions.create(model=chosen_model, messages=[{"role": "system", "content": "You are a real estate market data analyst."}, {"role": "user", "content": prompt}], temperature=0.2, max_tokens=900)
        return sanitize_for_pdf(response.choices[0].message.content.strip()) or fallback_narrative
    except Exception:
        return fallback_narrative

# ==============================================================================
# AUDIT-GRADE PDF EXPORT
# ==============================================================================
def build_pdf_report(subject: dict, reg_params: dict, grid_df: pd.DataFrame, narrative_text: str, trend_info: dict) -> BytesIO:
    clean_narrative = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', sanitize_for_pdf(narrative_text))
    min_val, max_val, midpoint = int(grid_df['indicated'].min()), int(grid_df['indicated'].max()), int((grid_df['indicated'].min() + grid_df['indicated'].max()) / 2)

    table_rows_html = ""
    for _, r in grid_df.iterrows():
        t_adj, g_adj, r_adj, o_adj, n_adj = f"${r['time_adj']:+,.0f}" if r['time_adj']!=0 else "$0", f"${r['gla_adj']:+,.0f}" if r['gla_adj']!=0 else "$0", f"${r['room_adj']:+,.0f}" if r['room_adj']!=0 else "$0", f"${(r['gar_adj']+r['bsmt_adj']+r['other_adj']):+,.0f}" if (r['gar_adj']+r['bsmt_adj']+r['other_adj'])!=0 else "$0", f"${r['net_adj']:+,.0f}"
        table_rows_html += f'<tr><td style="text-align: left; font-weight: bold; padding: 4px 3px;">{sanitize_for_pdf(r["address"])}<br/><span style="font-size: 6.5pt; color: #475569; font-weight: normal;">ID: {r["mls"]} | {r.get("days_old",0)} Days Old</span></td><td style="white-space: nowrap;">${r["price"]:,.0f}</td><td style="white-space: nowrap;">{r["contract_date"]}</td><td style="white-space: nowrap;">{int(r["gla"]):,} SF</td><td style="white-space: nowrap;">{r["rooms"]}</td><td style="white-space: nowrap;">{t_adj}</td><td style="white-space: nowrap;">{g_adj}</td><td style="white-space: nowrap;">{r_adj}</td><td style="white-space: nowrap;">{o_adj}</td><td style="white-space: nowrap;">{n_adj}<br/><span style="font-size: 6pt; color: #64748B;">({r["gross_pct"]}% gr / {r["net_pct"]}% net)</span></td><td style="white-space: nowrap; font-weight: bold; color: #1E3A8A; background-color: #F8FAFC;">${r["indicated"]:,.0f}</td></tr>'

    html = f"""
    <!DOCTYPE html><html><head><meta charset="utf-8"><style>
        @page {{ size: letter portrait; margin: 0.35in 0.4in; }}
        body {{ font-family: Helvetica, Arial, sans-serif; font-size: 8pt; line-height: 1.3; color: #1E293B; }}
        .header-bar {{ background-color: #1E3A8A; color: #ffffff; padding: 8px 12px; margin-bottom: 6px; }}
        .header-title {{ font-size: 13pt; font-weight: bold; margin: 0; }}
        .header-sub {{ font-size: 7.5pt; color: #E2E8F0; margin-top: 2px; }}
        .disclaimer {{ font-size: 6pt; color: #334155; background-color: #F8FAFC; padding: 5px 8px; border-left: 3px solid #1E3A8A; margin-bottom: 8px; }}
        .section-title {{ font-size: 9pt; font-weight: bold; color: #1E3A8A; border-bottom: 1px solid #CBD5E1; padding-bottom: 2px; margin-top: 8px; margin-bottom: 5px; }}
        .narrative-box {{ font-size: 7.5pt; line-height: 1.28; text-align: justify; margin-bottom: 8px; }}
        table.comp-grid {{ width: 100%; border-collapse: collapse; margin-top: 4px; margin-bottom: 6px; font-size: 6.8pt; }}
        table.comp-grid th {{ background-color: #F1F5F9; color: #0F172A; font-weight: bold; padding: 4px 2px; border: 0.5pt solid #CBD5E1; text-align: center; }}
        table.comp-grid td {{ padding: 3px 2px; border: 0.5pt solid #E2E8F0; text-align: center; vertical-align: middle; }}
        .kpi-table {{ width: 100%; margin-bottom: 6px; }}
        .kpi-box {{ background-color: #F8FAFC; border: 0.5pt solid #CBD5E1; padding: 4px; text-align: center; }}
        .kpi-val {{ font-size: 10pt; font-weight: bold; color: #1E3A8A; }}
        .kpi-lbl {{ font-size: 6pt; color: #64748B; text-transform: uppercase; font-weight: bold; }}
    </style></head>
    <body>
        <div class="header-bar">
            <div class="header-title">Automated Market Range & Listing Defense Brief ({subject['val_mode']})</div>
            <div class="header-sub">Subject: {sanitize_for_pdf(subject['address'])} | County: {subject['county']} | Type: {subject['property_type']} | Date: {subject['val_date']}</div>
        </div>
        <div class="disclaimer"><strong>IMPORTANT NOTICE:</strong> THIS IS NOT AN APPRAISAL. This report is an automated CMA for consultation purposes. Not developed in accordance with USPAP.</div>
        <table class="kpi-table"><tr>
            <td class="kpi-box"><div class="kpi-lbl">Statistical Benchmark</div><div class="kpi-val">${reg_params['benchmark']:,.0f}</div></td>
            <td class="kpi-box"><div class="kpi-lbl">Indicated Midpoint</div><div class="kpi-val">${midpoint:,.0f}</div></td>
            <td class="kpi-box"><div class="kpi-lbl">Indicated Value Range</div><div class="kpi-val">${min_val:,.0f} - ${max_val:,.0f}</div></td>
            <td class="kpi-box"><div class="kpi-lbl">{subject['property_type']} Mkt Trend</div><div class="kpi-val">{trend_info['monthly_pct']:+.2f}% / mo</div></td>
        </tr></table>
        <div class="section-title">1. Executive Pricing Strategy & Market Narrative</div>
        <div class="narrative-box">{clean_narrative}</div>
        <div class="section-title">2. Sales Comparison Model Grid (Strict {subject['property_type']} Matching)</div>
        <table class="comp-grid">
            <thead><tr>
                <th style="width: 23%; text-align: left;">Comparable Property</th><th style="width: 8%;">Price</th><th style="width: 7%;">Date</th><th style="width: 7%;">GLA</th>
                <th style="width: 7%;">Beds/Baths</th><th style="width: 7%;">Time Adj</th><th style="width: 7%;">GLA Adj</th><th style="width: 7%;">Room Adj</th>
                <th style="width: 8%;">Gar/Bsmt/Age</th><th style="width: 9%;">Net / Gross</th><th style="width: 10%;">Indicated</th>
            </tr></thead>
            <tbody>{table_rows_html}</tbody>
        </table>
        <div style="font-size: 6.5pt; color: #64748B; margin-top: 4px;">*Model Parameters: Strict {subject['property_type']} matching applied. Extracted GLA Rate: ${reg_params['gla_rate']}/SF. Market condition trend derived from historical sales ({trend_info['monthly_pct']:+.2f}%/mo, n={trend_info['sample_size']}). Garage valued at $10,000/bay; net finished basement differential at $50/SF.</div>
    </body></html>
    """
    pdf_buffer = BytesIO()
    pisa.CreatePDF(html, dest=pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer

# ==============================================================================
# UI COMPONENTS
# ==============================================================================
if "active_comps_df" not in st.session_state: st.session_state["active_comps_df"] = None
if "autofill_subject" not in st.session_state: st.session_state["autofill_subject"] = {}

with st.sidebar:
    st.header("1. County & Search Scope")
    selected_county = st.selectbox("NC County:", list(NC_COUNTY_CONFIG.keys()), index=0)
    st.caption(f"Market Profile: {NC_COUNTY_CONFIG[selected_county]['tier']}")
    search_radius = st.slider("Comparable Search Radius (Miles):", 0.25, 5.0, float(NC_COUNTY_CONFIG[selected_county]["default_radius"]), 0.25)

    st.header("2. Deliverable Strategy Mode")
    val_mode = st.radio("Brief Purpose:", ["Pre-Listing / CMA Strategy", "Executed Contract Defense"])
    val_date = st.date_input("Effective Analysis Date:", value=date.today())

    st.header("3. Market Conditions (Time Trend)")
    trend_mode = st.radio("Time Adjustment Method:", ["Auto-Detect from Historical Data", "Manual Override"])
    manual_trend_override = st.slider("Manual Monthly Trend (%):", -0.75, 0.75, 0.0, 0.05) if trend_mode == "Manual Override" else 0.0

    st.header("4. Comparable Selection Mode")
    comp_mode = st.radio("Selection Strategy:", ["Algorithmic (Smart Limits)", "Manual Override (Pick Specific Comps)"])

    st.header("5. Narrative Engine")
    groq_api_key = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
    if groq_api_key: st.success("AI Narrative Engine Connected")
    else: st.info("Running Deterministic Narrative Mode")

st.markdown('<div class="report-header">8-County NC Valuation & Listing Defense Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Strict Property Type Segmentation • Elastic 12-24 Month Limits • Historical Trend Extraction</div>', unsafe_allow_html=True)

tab_addr, tab_csv, tab_manual = st.tabs(["1. Address Lookup (Free Public Records)", "2. Upload MLS CSV (Realtors / Internal)", "3. Manual Comp Grid (Fallback)"])

with tab_addr:
    lookup_addr = st.text_input("Property Street Address, City, State, ZIP:", placeholder="e.g., 123 Main St, Raleigh, NC 27601")
    if st.button("Fetch Public Parcel & Comparable Data"):
        if not lookup_addr.strip(): st.warning("Please enter a property street address, city, and ZIP code first.")
        else:
            with st.spinner(f"Querying public parcel records within {search_radius} miles..."):
                try:
                    subj_auto, api_df = fetch_comps_from_nconemap(lookup_addr, search_radius)
                    if api_df.empty: st.error("No parcel sales records returned. Try increasing the Search Radius.")
                    else:
                        st.session_state["autofill_subject"], st.session_state["active_comps_df"] = subj_auto, api_df
                        st.success(f"Found {len(api_df)} records. Verify subject specs below.")
                except Exception as e: st.error(f"Public records lookup failed: {e}")

with tab_csv:
    uploaded_file = st.file_uploader("Upload MLS Search Export (.csv)", type=["csv"])
    if uploaded_file is not None:
        csv_df = load_and_clean_csv(uploaded_file)
        st.session_state["active_comps_df"] = csv_df
        st.success(f"Loaded {len(csv_df)} market records from MLS CSV.")

with tab_manual:
    empty_manual = pd.DataFrame(columns=["MLS #", "Address", "Clean_Property_Type", "Clean_Price", "Clean_GLA", "Clean_Bsmt_Fin", "Clean_Year", "Clean_Beds", "Clean_FB", "Clean_HB", "Clean_Subdivision", "Clean_Date", "Clean_Garage"])
    edited_manual_df = st.data_editor(
        empty_manual, num_rows="dynamic", use_container_width=True,
        column_config={
            "MLS #": st.column_config.TextColumn("ID", default="COMP"), "Address": st.column_config.TextColumn("Address"),
            "Clean_Property_Type": st.column_config.SelectboxColumn("Type", options=["Detached", "Attached", "Condo"], default="Detached"),
            "Clean_Price": st.column_config.NumberColumn("Sold Price", min_value=0, step=1000), "Clean_GLA": st.column_config.NumberColumn("GLA", min_value=0, step=10),
            "Clean_Bsmt_Fin": st.column_config.NumberColumn("Fin. Bsmt", min_value=0, step=50, default=0), "Clean_Year": st.column_config.NumberColumn("Year Built", default=2000),
            "Clean_Beds": st.column_config.NumberColumn("Beds", default=3), "Clean_FB": st.column_config.NumberColumn("F Baths", default=2), "Clean_HB": st.column_config.NumberColumn("H Baths", default=0),
            "Clean_Subdivision": st.column_config.TextColumn("Subdivision", default=""), "Clean_Date": st.column_config.DateColumn("Close Date"), "Clean_Garage": st.column_config.NumberColumn("Garage Bays", default=2),
        }
    )
    if st.button("Use Manual Comps Table"):
        if edited_manual_df.empty: st.warning("Add at least 3 comparable rows to the table first.")
        else:
            valid_manual = edited_manual_df[pd.to_numeric(edited_manual_df["Clean_Price"], errors="coerce") > 20000].copy()
            valid_manual["Clean_Date"] = pd.to_datetime(valid_manual["Clean_Date"], errors="coerce")
            for col, dval in [("Clean_Bsmt_Fin", 0.0), ("Clean_Year", 2000), ("Clean_Beds", 3), ("Clean_FB", 2), ("Clean_HB", 0), ("Clean_Garage", 2.0)]: valid_manual[col] = pd.to_numeric(valid_manual[col], errors="coerce").fillna(dval)
            valid_manual["Clean_Subdivision"] = valid_manual["Clean_Subdivision"].fillna("").astype(str)
            valid_manual["Clean_Property_Type"] = valid_manual["Clean_Property_Type"].fillna("Detached").astype(str)
            st.session_state["active_comps_df"] = valid_manual
            st.success(f"Loaded {len(valid_manual)} manual comparables.")

st.divider()
st.subheader("Subject Property Parameters")
af = st.session_state.get("autofill_subject", {})

col1, col2, col3 = st.columns(3)
with col1:
    subj_address = st.text_input("Subject Address:", value=af.get("address", ""))
    subj_prop_type = st.selectbox("Property Type:", ["Detached", "Attached", "Condo"], index=["Detached", "Attached", "Condo"].index(af.get("property_type", "Detached")))
    subj_subdivision = st.text_input("Subdivision (Leave blank for radius):", value=af.get("subdivision", ""))
    subj_year = st.number_input("Year Built:", min_value=1800, max_value=2030, value=int(af.get("year_built", 2000)), step=1)
    subj_beds = st.number_input("Above-Grade Bedrooms:", min_value=0, max_value=15, value=int(af.get("beds", 3)), step=1)

with col2:
    subj_gla = st.number_input("Above-Grade Measured GLA (SF):", min_value=0, max_value=25000, value=int(af.get("gla", 0)), step=10)
    subj_bsmt_fin = st.number_input("Below-Grade Finished Basement (SF):", min_value=0, max_value=15000, value=0, step=50)
    subj_fb = st.number_input("Above-Grade Full Baths:", min_value=0, max_value=15, value=int(af.get("baths_full", 2)), step=1)
    subj_hb = st.number_input("Above-Grade Half Baths:", min_value=0, max_value=10, value=int(af.get("baths_half", 0)), step=1)

with col3:
    subj_condition = st.selectbox("Condition Rating:", ["Average", "Good", "Excellent", "Fair"])
    subj_garage = st.number_input("Garage Bays:", min_value=0, max_value=10, value=2, step=1)
    subj_fireplaces = st.number_input("Fireplaces:", min_value=0, max_value=10, value=0, step=1)

st.markdown("**Exterior Features & Amenities:**")
am_cols = st.columns(6)
with am_cols[0]: has_front_porch = st.checkbox("Front Porch", value=False)
with am_cols[1]: has_screen_porch = st.checkbox("Screen Porch", value=False)
with am_cols[2]: has_deck = st.checkbox("Deck", value=False)
with am_cols[3]: has_patio = st.checkbox("Patio", value=False)
with am_cols[4]: has_fence = st.checkbox("Fenced Yard", value=False)
with am_cols[5]: has_pool = st.checkbox("Pool", value=False)

active_df = st.session_state.get("active_comps_df")

if active_df is not None and not active_df.empty:
    subject_data = {
        "address": subj_address.strip(), "county": af.get("county", selected_county), "property_type": subj_prop_type,
        "subdivision": subj_subdivision.strip(), "year_built": subj_year, "gla": subj_gla, "beds": subj_beds,
        "baths_full": subj_fb, "baths_half": subj_hb, "garage_bays": subj_garage, "condition": subj_condition,
        "fireplaces": subj_fireplaces, "bsmt_fin": subj_bsmt_fin, "val_date": str(val_date), "val_mode": val_mode,
        "amenities": {"Front Porch": has_front_porch, "Screen Porch": has_screen_porch, "Deck": has_deck, "Patio": has_patio, "Fenced Yard": has_fence, "Pool": has_pool}
    }

    trend_info = calculate_historical_market_trend(active_df, subject_data) if trend_mode == "Auto-Detect from Historical Data" else {"monthly_pct": manual_trend_override, "annual_pct": round(manual_trend_override * 12.0, 1), "sample_size": len(active_df[active_df['Clean_Property_Type'] == subj_prop_type]), "span_days": 0, "status": f"Manual override: {manual_trend_override:+.2f}%/mo."}
    subject_data["monthly_trend_pct"] = trend_info["monthly_pct"]
    st.info(f"📈 **Market Conditions Analysis:** {trend_info['status']}")

    manual_ids = []
    if comp_mode == "Manual Override (Pick Specific Comps)":
        mls_options = active_df[active_df['Clean_Property_Type'] == subj_prop_type]['MLS #'].tolist()
        format_func = lambda x: f"{x} - {active_df[active_df['MLS #']==x]['Address'].values[0]} (${active_df[active_df['MLS #']==x]['Clean_Price'].values[0]:,.0f})"
        manual_ids = st.multiselect(f"Choose 3 to 4 {subj_prop_type} Comparables:", mls_options, format_func=format_func)

    if st.button("Generate Market Range & Defense Brief", type="primary", use_container_width=True):
        if not subject_data["address"] or subject_data["gla"] <= 100: st.error("Please enter a valid Subject Address and Measured GLA (SF).")
        else:
            reg_params = run_regression_analysis(active_df, subject_data)
            mode_key = "manual" if comp_mode == "Manual Override (Pick Specific Comps)" else "algorithmic"
            comps_df = select_comparables_with_subdivision(active_df, subject_data, reg_params, mode=mode_key, manual_mls_ids=manual_ids)

            if comps_df.empty:
                st.error(f"No valid {subj_prop_type} comparables found within the past 24 months. Try expanding your search radius.")
            else:
                grid_df = calculate_grid_adjustments(comps_df, subject_data, reg_params)
                with st.spinner("Compiling market narrative..."): narrative = generate_narrative_groq(groq_api_key, subject_data, reg_params, grid_df, trend_info)

                min_ind, max_ind, mid_ind = int(grid_df['indicated'].min()), int(grid_df['indicated'].max()), int((grid_df['indicated'].min() + grid_df['indicated'].max()) / 2)
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Hedonic Benchmark", f"${reg_params['benchmark']:,.0f}")
                k2.metric("Indicated Midpoint", f"${mid_ind:,.0f}")
                k3.metric("Indicated Range", f"${min_ind:,.0f} - ${max_ind:,.0f}")
                k4.metric("Extracted Market Trend", f"{trend_info['monthly_pct']:+.2f}% / mo")

                time_caveat_header = " (<12 Mos Preference Applied)" if int(grid_df.get('days_old', pd.Series([0])).max()) <= 365 else " (Expanded to 24 Mos Due to Low Volume)"
                st.markdown(f"### Comparable Sales Adjustment Grid (Strict {subj_prop_type}){time_caveat_header}")
                st.dataframe(grid_df[['address', 'subdivision', 'contract_date', 'price', 'gla', 'rooms', 'time_adj', 'gla_adj', 'room_adj', 'gar_adj', 'bsmt_adj', 'other_adj', 'net_adj', 'gross_pct', 'indicated']], use_container_width=True)

                with st.expander("Preview Executive Market Narrative", expanded=True): st.write(narrative, unsafe_allow_html=True)
                st.download_button(label="📥 Download Market Range & Defense Brief (PDF)", data=build_pdf_report(subject_data, reg_params, grid_df, narrative, trend_info), file_name=f"Market_Brief_{re.sub(r'[^a-zA-Z0-9]', '_', subj_address)}.pdf", mime="application/pdf", use_container_width=True)
else:
    st.info("Use Tab 1 (Free Public Records Lookup), Tab 2 (Upload MLS CSV), or Tab 3 (Manual Comp Grid) above to load comparable sales.")
