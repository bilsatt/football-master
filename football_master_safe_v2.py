import os
import streamlit as st
import requests
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta, timezone

try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

try:
    from model_plus_dc import predict_match, implied_from_odds
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    MODEL_ERR = str(e)

def _get_secret(name: str, default: str = "") -> str:
    try:
        val = st.secrets.get(name, None)
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(name, default)

st.set_page_config(page_title="⚽ Football Master v5d — Cloud + O/U", layout="wide")
st.title("⚽ Football Master v5d — DC + Opening PDF + OddAlerts + DrawGuard/CLV + Over/Under")
st.caption("Yol 1 (bulut) için hazır. Ekstra: Over 1.5 ve Under 2.5 olasılıkları + Excel'e yazım.")

with st.sidebar:
    st.header("🔐 Anahtarlar")
    default_api = _get_secret("APIFOOTBALL_KEY", "")
    default_oa  = _get_secret("ODDALERTS_TOKEN", "")
    API_KEY = st.text_input("API-Football PRO Key", type="password", value=default_api)
    OA_TOKEN = st.text_input("OddAlerts API Token (opsiyonel)", type="password", value=default_oa)

    st.header("⚙️ Özellikler")
    feat_global_search = st.checkbox("GLOBAL Takım Arama (Lig Otomatik)", True)
    feat_league_flow   = st.checkbox("Klasik: Lig Seç → Takım", True)
    feat_opening_pdf   = st.checkbox("Açılış Oranları: PDF/Manuel", True)
    feat_oddalerts     = st.checkbox("OddAlerts Opening/Closing/Peak", True if (OA_TOKEN or default_oa) else False)
    feat_batch_roi     = st.checkbox("Batch + ROI", True)

    st.header("🔧 Model & Bahis Ayarları")
    season = st.number_input("Sezon (başlangıç yılı)", min_value=2015, max_value=2026, value=2023, step=1)
    alpha = st.slider("Odds / Model Harmanı (α)", 0.0, 1.0, 0.80, 0.05)
    max_goals = st.slider("DC Matrisi Maks Gol", 4, 8, 6)
    min_conf = st.slider("No Bet Eşiği (max olasılık)", 0.40, 0.70, 0.50, 0.01)
    stake = st.number_input("Bahis Stake (Batch)", min_value=10.0, value=100.0, step=10.0)
    kelly_fraction = st.selectbox("Bahis Yöntemi (Batch)", ["Flat", "Kelly 1/2", "Kelly 1/4"], index=2)

    st.header("🎰 Oran Seçenekleri (API-Football)")
    bet_name = st.selectbox("Market", ["Match Winner"], index=0)
    bm_filter_text = st.text_input("Bookmaker filtresi (virgülle ayır) • boş = hepsi", value="")

    if feat_opening_pdf:
        st.subheader("📄 Açılış Oranları (Opsiyonel)")
        pdf_file = st.file_uploader("PDF yükle", type=["pdf"])
        manual_open = st.text_input("Manuel Açılış (1X2): 1.42,3.57,5.11", value="")

    st.header("🛡️ Draw-Guard / CLV")
    draw_guard = st.slider("Beraberlik Koruma Eşiği p(D)", 0.20, 0.45, 0.30, 0.01)
    switch_choice = st.selectbox("p(D) eşiği aşılınca otomatik pazar", ["1X Double Chance", "Home DNB (0)", "None"], index=0)
    clv_guard = st.slider("CLV uyarı eşiği (Now vs Open, % düşüş sınırı)", -15.0, 0.0, -5.0, 0.5)

if not API_KEY:
    st.warning("API anahtarı boş. Soldan API-Football PRO Key girin ya da Secrets/Env olarak 'APIFOOTBALL_KEY' tanımlayın.")
    st.stop()
headers = {"x-apisports-key": API_KEY}

# --- Bu sürümde sadece güvenli 'Lig Seç → Takım' bölümü bırakıldı (test için kısaltılmış) ---

st.subheader("Takım Seçimi (Güvenli Mod)")

leagues_df = pd.DataFrame([
    {"id": 39, "name": "Premier League", "country": "England"},
    {"id": 140, "name": "La Liga", "country": "Spain"}
])
sel_league_name = st.selectbox("Lig", leagues_df.apply(lambda r: f"{r['country']} - {r['name']}", axis=1).tolist())
league_id = int(leagues_df[leagues_df.apply(lambda r: f"{r['country']} - {r['name']}", axis=1) == sel_league_name]["id"].iloc[0])

teams_df = pd.DataFrame([
    {"id": 33, "name": "Manchester United"},
    {"id": 34, "name": "Liverpool"}
])

team_filter = st.text_input("🔎 Takım Ara (listeyi filtrele)", value="").strip().lower()

if team_filter and "name" in teams_df.columns:
    tmask = teams_df["name"].str.lower().str.contains(team_filter, na=False)
    filtered = teams_df[tmask].reset_index(drop=True)
    if filtered.empty:
        filtered = teams_df.copy()
else:
    filtered = teams_df.copy()

if filtered.empty or "name" not in filtered.columns:
    st.warning("⚠️ Takım verisi alınamadı veya 'name' sütunu bulunamadı. Lütfen API anahtarını veya lig seçimini kontrol et.")
    st.stop()

col1, col2 = st.columns(2)
home_team = col1.selectbox("Ev Sahibi", filtered["name"].tolist(), index=0 if not filtered.empty else None)
away_team = col2.selectbox("Deplasman", filtered["name"].tolist(), index=1 if len(filtered) > 1 else 0)

st.success(f"Seçilen maç: {home_team} vs {away_team}")
