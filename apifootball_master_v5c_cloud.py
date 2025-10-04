
import os
import streamlit as st
import requests
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta, timezone

# Optional deps
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

# Model
try:
    from model_plus_dc import predict_match, implied_from_odds
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    MODEL_ERR = str(e)

# -------- Secrets helpers (cloud-ready) --------
def _get_secret(name: str, default: str = "") -> str:
    # Try Streamlit Secrets first
    try:
        val = st.secrets.get(name, None)  # streamlit cloud
        if val:
            return str(val)
    except Exception:
        pass
    # Fallback: environment variable (HF Spaces / other PaaS)
    return os.environ.get(name, default)

st.set_page_config(page_title="⚽ Football Master v5c — Cloud Ready", layout="wide")
st.title("⚽ Football Master v5c — DC + Opening PDF + OddAlerts + Odds Fallback + DrawGuard/CLV (Cloud Ready)")
st.caption("Yol 1: Streamlit Cloud/HF Spaces için hazır; API anahtarlarını **Secrets/Env**'den otomatik alır.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("🔐 Anahtarlar")
    # Secrets/Env (otomatik)
    default_api = _get_secret("APIFOOTBALL_KEY", "")
    default_oa  = _get_secret("ODDALERTS_TOKEN", "")
    # Kullanıcı isterse override edebilir
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

# ---------------- Utils ----------------
if not API_KEY:
    st.warning("API anahtarı boş. Soldan **API-Football PRO Key** girin ya da Secrets/Env olarak 'APIFOOTBALL_KEY' tanımlayın.")
    st.stop()
headers = {"x-apisports-key": API_KEY}

def to_utc(dt_str: str) -> datetime:
    if not dt_str: return datetime.now(timezone.utc)
    s = dt_str.replace("Z", "+00:00")
    try: dt = datetime.fromisoformat(s)
    except Exception:
        try: dt = datetime.fromisoformat(s.split("T")[0]+"T00:00:00+00:00")
        except Exception: return datetime.now(timezone.utc)
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    else: dt = dt.astimezone(timezone.utc)
    return dt

def api_get(endpoint, params=None):
    url = f"https://v3.football.api-sports.io/{endpoint}"
    r = requests.get(url, headers=headers, params=params, timeout=30)
    try:
        r.raise_for_status()
    except Exception as e:
        st.error(f"HTTP Hatası: {e}\nYanıt: {r.text[:300]}")
        raise
    return r.json()

@st.cache_data(show_spinner=False)
def leagues_list():
    data = api_get("leagues")
    out = [{"id": it["league"]["id"], "name": it["league"]["name"], "country": it["country"]["name"]}
           for it in data.get("response", [])]
    return pd.DataFrame(out)

@st.cache_data(show_spinner=False)
def teams_in(league_id, season):
    data = api_get("teams", {"league": league_id, "season": season})
    out = [{"id": t["team"]["id"], "name": t["team"]["name"]} for t in data.get("response", [])]
    return pd.DataFrame(out)

@st.cache_data(show_spinner=False)
def teams_search(query):
    if not query or len(query.strip()) < 2:
        return pd.DataFrame()
    data = api_get("teams", {"search": query.strip()})
    out = []
    for t in data.get("response", []):
        out.append({
            "id": t["team"]["id"],
            "name": t["team"]["name"],
            "country": t.get("team",{}).get("country") or t.get("country",{}).get("name") or "",
        })
    return pd.DataFrame(out).drop_duplicates(subset=["id"]).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def leagues_by_team(team_id, season):
    data = api_get("leagues", {"team": team_id, "season": season})
    out = [{
        "id": it["league"]["id"],
        "name": it["league"]["name"],
        "country": it["country"]["name"],
        "type": it["league"].get("type")
    } for it in data.get("response", [])]
    return pd.DataFrame(out)

@st.cache_data(show_spinner=False)
def fixtures_by_team(team_id, season):
    data = api_get("fixtures", {"team": team_id, "season": season})
    return data.get("response", [])

@st.cache_data(show_spinner=False)
def fixtures_by_league(league_id, season, status=None):
    params = {"league": league_id, "season": season}
    if status: params["status"] = status
    data = api_get("fixtures", params)
    return data.get("response", [])

def last5_finished(team_id):
    fx = fixtures_by_team(team_id, season)
    finished = [f for f in fx if f["fixture"]["status"]["short"] in ("FT","AET","PEN")]
    finished.sort(key=lambda x: to_utc(x["fixture"]["date"]), reverse=True)
    last5 = finished[:5]
    rows = []
    w=d=l=0; gf=ga=0; cnt=0
    for f in last5:
        hg = f["score"]["fulltime"]["home"]; ag = f["score"]["fulltime"]["away"]
        if hg is None or ag is None: continue
        cnt += 1
        is_home = (f["teams"]["home"]["id"] == team_id)
        gf += (hg if is_home else ag)
        ga += (ag if is_home else hg)
        if (hg>ag and is_home) or (ag>hg and not is_home): w+=1
        elif hg==ag: d+=1
        else: l+=1
        rows.append({"Tarih": f["fixture"]["date"], "Ev": f["teams"]["home"]["name"], "Dep": f["teams"]["away"]["name"], "Skor": f"{hg}-{ag}", "Durum": f["fixture"]["status"]["short"]})
    return {"G":w,"B":d,"M":l,"GF":round(gf/max(cnt,1),2),"GA":round(ga/max(cnt,1),2)}, pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def odds_for_fixture_or_league(fixture_id=None, league_id=None, season=None, home_id=None, away_id=None, bet_name="Match Winner", bm_filters=None):
    # 1) Fixture-scoped
    if fixture_id:
        data = api_get("odds", {"fixture": fixture_id})
        resp = data.get("response", [])
        rows = _collect_odds_rows(resp, bet_name, bm_filters)
        if rows: return pd.DataFrame(rows), "fixture"
    # 2) Fallback: league+season
    if league_id and season:
        data2 = api_get("odds", {"league": league_id, "season": season})
        resp2 = data2.get("response", [])
        rows2 = _collect_odds_rows(resp2, bet_name, bm_filters, want_home_id=home_id, want_away_id=away_id)
        if rows2: return pd.DataFrame(rows2), "league"
    return pd.DataFrame(), None

def _collect_odds_rows(resp, bet_name, bm_filters, want_home_id=None, want_away_id=None):
    rows = []
    bm_filters_set = None
    if bm_filters:
        bm_filters_set = {b.strip().lower() for b in bm_filters.split(",") if b.strip()}
    for item in resp or []:
        try:
            t_home = item["teams"]["home"]["id"]
            t_away = item["teams"]["away"]["id"]
            if want_home_id and want_away_id:
                if not (t_home == want_home_id and t_away == want_away_id):
                    continue
            for bm in item.get("bookmakers", []):
                if bm_filters_set and bm.get("name","").lower() not in bm_filters_set:
                    continue
                for bet in bm.get("bets", []):
                    if bet.get("name") == bet_name:
                        for val in bet.get("values", []):
                            rows.append({"Bookmaker": bm["name"], "Sonuç": val["value"], "Oran": float(val["odd"])})
        except Exception:
            continue
    return rows

# -------- Opening PDF helpers --------
def _num(s):
    if s is None: return None
    s = str(s).strip().replace(",", ".")
    try: return float(s)
    except: return None

def _clean_team(s):
    if s is None: return ""
    return " ".join(str(s).split()).strip()

def parse_opening_pdf(file_obj):
    if pdfplumber is None:
        st.error("pdfplumber yüklü değil. pip install pdfplumber")
        return pd.DataFrame()
    rows = []
    with pdfplumber.open(file_obj) as pdf:
        for p in pdf.pages:
            try:
                tables = p.extract_tables()
            except Exception:
                tables = []
            for tbl in tables or []:
                for r in tbl:
                    if not r: continue
                    nums = [_num(x) for x in r]
                    odds = [x for x in nums if x and 1.01 <= x <= 50.0]
                    if len(odds) >= 3:
                        txts = [c for c in r if c and not str(c).replace(",",".").replace(".","",1).isdigit()]
                        home=""; away=""
                        joined = " ".join([_clean_team(t) for t in txts])
                        if "-" in joined:
                            parts = joined.split("-")
                            if len(parts)>=2:
                                home = _clean_team(parts[0]); away = _clean_team("-".join(parts[1:]))
                        else:
                            toks = [t for t in txts if len(t.strip())>1]
                            if len(toks)>=2:
                                home=_clean_team(toks[0]); away=_clean_team(toks[1])
                        if home or away:
                            rows.append({"home": home, "away": away, "open_home": odds[0], "open_draw": odds[1], "open_away": odds[2]})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["open_home","open_draw","open_away"], how="any").drop_duplicates()
    return df

def match_opening_for(home_name, away_name, opening_df):
    if opening_df is None or opening_df.empty: return None
    if fuzz is None:
        hmask = opening_df["home"].str.contains(home_name, case=False, na=False) & opening_df["away"].str.contains(away_name, case=False, na=False)
        if not opening_df[hmask].empty:
            row = opening_df[hmask].iloc[0]
            return {"Home": row["open_home"], "Draw": row["open_draw"], "Away": row["open_away"]}
        return None
    best = None; best_score = -1
    for _, r in opening_df.iterrows():
        s1 = (fuzz.token_sort_ratio(home_name, r["home"]) + fuzz.token_sort_ratio(away_name, r["away"])) / 2.0
        s2 = (fuzz.token_sort_ratio(home_name, r["away"]) + fuzz.token_sort_ratio(away_name, r["home"])) / 2.0
        if s1 > best_score: best_score = s1; best = r
        if s2 > best_score: best_score = s2; best = r
    if best is not None and best_score >= 75:
        return {"Home": best["open_home"], "Draw": best["open_draw"], "Away": best["open_away"]}
    return None

def implied(dec):
    try:
        return 1.0/float(dec) if dec else None
    except Exception:
        return None

# --- Gol olasılıkları (Poisson) ---
def calc_goal_probs(lam_h: float, lam_a: float):
    p_home = 1 - np.exp(-lam_h)                 # Ev gol atar (>=1)
    p_away = 1 - np.exp(-lam_a)                 # Dep gol atar (>=1)
    p_btts = 1 - np.exp(-lam_h) - np.exp(-lam_a) + np.exp(-(lam_h + lam_a))
    p_over05 = 1 - np.exp(-(lam_h + lam_a))     # Toplam gol > 0.5
    return p_home, p_away, p_btts, p_over05

# OddAlerts helpers
def oa_enabled(token): return bool(token and token.strip())

def oa_get(path, params=None):
    if not oa_enabled(OA_TOKEN): return {}
    params = params or {}; params["api_token"] = OA_TOKEN
    url = f"https://data.oddalerts.com/api{path}"
    r = requests.get(url, params=params, headers={"User-Agent":"FootballMaster/1.0","Accept":"application/json"}, timeout=30)
    if r.status_code != 200:
        return {}
    try:
        return r.json()
    except Exception:
        return {}

def oa_find_fixture_id(home_name, away_name, start=None, end=None):
    if not oa_enabled(OA_TOKEN): return None, None
    if start is None: start = datetime.now(timezone.utc) - timedelta(days=3)
    if end   is None: end   = datetime.now(timezone.utc) + timedelta(days=10)
    def unix(dt): 
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    js = oa_get("/fixtures/between", {"from": unix(start), "to": unix(end), "include":"stats"})
    data = js.get("data", []) if isinstance(js, dict) else []
    best = (None, -1)
    for f in data:
        h = f.get("home", {}).get("name",""); a = f.get("away", {}).get("name","")
        sc = (fuzz.token_sort_ratio(home_name, h) + fuzz.token_sort_ratio(away_name, a))/2 if fuzz else int(home_name.lower() in h.lower() and away_name.lower() in a.lower())*100
        if sc > best[1]: best = (f, sc)
    if best[0] is None or best[1] < 70: return None, None
    return best[0]["id"], best[0]

def oa_odds_history(fid, market_id=4, bookmaker_id=1):
    if not oa_enabled(OA_TOKEN) or fid is None: return {}
    js = oa_get(f"/odds/history/{fid}", {"markets": market_id, "bookmakers": bookmaker_id})
    return js.get("data", {}) if isinstance(js, dict) else {}

# ---------------- UI ----------------
tabs = ["Tek Maç Analiz"]
feat_batch_roi_default = True
if feat_batch_roi:
    tabs.append("Batch + ROI")
panes = st.tabs(tabs)

with panes[0]:
    st.subheader("Takım Seçimi")
    home_id = away_id = None
    home_team = away_team = None
    league_id = None
    selected_comp_label = None

    mode_opts = []
    if feat_league_flow:   mode_opts.append("Lig Seç → Takım")
    if feat_global_search: mode_opts.append("GLOBAL Takım Arama (Lig Otomatik)")
    mode = st.radio("Yöntem", options=mode_opts, horizontal=True)

    def leagues_list_local():
        df = leagues_list()
        return df

    if mode == "Lig Seç → Takım":
        leagues_df = leagues_list_local()
        sel_league_name = st.selectbox("Lig", leagues_df.apply(lambda r: f"{r['country']} - {r['name']}", axis=1).tolist())
        league_id = int(leagues_df[leagues_df.apply(lambda r: f"{r['country']} - {r['name']}", axis=1)==sel_league_name]["id"].iloc[0])
        ctop = st.columns([4,2])
        team_filter = ctop[1].text_input("🔎 Takım Ara (listeyi filtrele)", value="").strip().lower()
        teams_df = teams_in(league_id, season)
        if team_filter:
            tmask = teams_df["name"].str.lower().str.contains(team_filter, na=False)
            filtered = teams_df[tmask].reset_index(drop=True)
            if filtered.empty: filtered = teams_df.copy()
        else:
            filtered = teams_df.copy()
        col1, col2 = st.columns(2)
        home_team = col1.selectbox("Ev Sahibi", filtered["name"].tolist(), index=0 if not filtered.empty else None)
        away_team = col2.selectbox("Deplasman", filtered["name"].tolist(), index=1 if len(filtered)>1 else 0)
        home_id = int(teams_df[teams_df["name"]==home_team]["id"].iloc[0])
        away_id = int(teams_df[teams_df["name"]==away_team]["id"].iloc[0])
        selected_comp_label = sel_league_name

    else:
        g1, g2 = st.columns(2)
        q_home = g1.text_input("Ev Sahibi (isim yaz)", value="")
        q_away = g2.text_input("Deplasman (isim yaz)", value="")
        if len(q_home.strip()) >= 2:
            dfh = teams_search(q_home)
            if not dfh.empty:
                opt_h = dfh.apply(lambda r: f"{r['name']} ({r['country']}) — id:{r['id']}", axis=1).tolist()
                ch = g1.selectbox("Ev Takım Seç", opt_h, index=0)
                idx = opt_h.index(ch); home_id = int(dfh.iloc[idx]["id"]); home_team = dfh.iloc[idx]["name"]
        if len(q_away.strip()) >= 2:
            dfa = teams_search(q_away)
            if not dfa.empty:
                opt_a = dfa.apply(lambda r: f"{r['name']} ({r['country']}) — id:{r['id']}", axis=1).tolist()
                ca = g2.selectbox("Deplasman Takım Seç", opt_a, index=0)
                idx = opt_a.index(ca); away_id = int(dfa.iloc[idx]["id"]); away_team = dfa.iloc[idx]["name"]
        if home_id and away_id:
            lh = leagues_by_team(home_id, season)
            la = leagues_by_team(away_id, season)
            if not lh.empty and not la.empty:
                common = pd.merge(lh, la, on=["id","name","country","type"], how="inner")
                if not common.empty:
                    leagues_ordered = pd.concat([common[common["type"]=="League"], common[common["type"]!="League"]])
                    opt = leagues_ordered.apply(lambda r: f"{r['country']} - {r['name']} ({r['type']})", axis=1).tolist()
                    selected_comp_label = st.selectbox("Müsabaka (ortak)", opt, index=0)
                    league_id = int(leagues_ordered.iloc[opt.index(selected_comp_label)]["id"])

    # Run
    if (home_id and away_id) and st.button("Analiz & Tahmin", key="go_single"):
        # Last 5
        hf, hf_df = last5_finished(home_id); af, af_df = last5_finished(away_id)
        c3, c4 = st.columns(2)
        c3.metric(f"{home_team} G-B-M", f"{hf['G']}-{hf['B']}-{hf['M']}"); c3.metric(f"{home_team} GF/GA", f"{hf['GF']}/{hf['GA']}")
        c4.metric(f"{away_team} G-B-M", f"{af['G']}-{af['B']}-{af['M']}"); c4.metric(f"{away_team} GF/GA", f"{af['GF']}/{af['GA']}")
        with st.expander(f"{home_team} • Son 5"): st.dataframe(hf_df if not hf_df.empty else pd.DataFrame([{"Bilgi":"Veri bulunamadı"}]))
        with st.expander(f"{away_team} • Son 5"): st.dataframe(af_df if not af_df.empty else pd.DataFrame([{"Bilgi":"Veri bulunamadı"}]))

        # Fixture & odds (with fallback)
        fx_home = fixtures_by_team(home_id, season)
        fixture = None
        for f in fx_home:
            if f["teams"]["home"]["id"]==home_id and f["teams"]["away"]["id"]==away_id:
                fixture = f; break
        fixture_id = fixture["fixture"]["id"] if fixture else None
        league_for_odds = fixture["league"]["id"] if fixture else league_id

        odds_df, source = odds_for_fixture_or_league(
            fixture_id=fixture_id,
            league_id=league_for_odds,
            season=season,
            home_id=home_id, away_id=away_id,
            bet_name=bet_name,
            bm_filters=bm_filter_text
        )
        st.subheader("📅 Oranlar")
        if fixture:
            st.info(f"Maç: {fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']} • {fixture['fixture']['date']}")
        if not odds_df.empty:
            if source == "league": st.caption("Not: Fixture bazında oran yoktu; Lig+Sezon fallback kullanıldı.")
            st.dataframe(odds_df)
            pvt = odds_df.pivot_table(index="Sonuç", values="Oran", aggfunc=np.median).to_dict()["Oran"]
            normalize = {"Home":"Home","1":"Home","home":"Home","Draw":"Draw","X":"Draw","draw":"Draw","Away":"Away","2":"Away","away":"Away"}
            odds_map = {}; 
            for k,v in pvt.items():
                kk = normalize.get(str(k), None)
                if kk: odds_map[kk] = float(v)
        else:
            st.warning("Bu maç için oran bulunamadı.")
            odds_map = None

        # Opening PDF / Manual
        open_map = None
        if feat_opening_pdf:
            opening_df = pd.DataFrame()
            if pdf_file is not None:
                opening_df = parse_opening_pdf(pdf_file)
                if not opening_df.empty:
                    st.success(f"PDF'den {len(opening_df)} satır açılış okundu.")
                    with st.expander("PDF Açılış Satırları (ilk 50)"): st.dataframe(opening_df.head(50))
                    open_map = match_opening_for(home_team, away_team, opening_df)
            if (open_map is None) and 'manual_open' in locals() and manual_open:
                try:
                    parts = [p.strip() for p in manual_open.replace(";",",").split(",")]
                    if len(parts)>=3:
                        open_map = {"Home": float(parts[0].replace(",", ".")), "Draw": float(parts[1].replace(",", ".")), "Away": float(parts[2].replace(",", "."))}
                except Exception:
                    pass
            if open_map:
                st.info(f"**Açılış Oranları** — 1:{open_map['Home']}  X:{open_map['Draw']}  2:{open_map['Away']}")

        # OddAlerts
        oa_open = oa_now = oa_peak = {}
        if feat_oddalerts and (OA_TOKEN or default_oa):
            fid, fo = oa_find_fixture_id(home_team, away_team)
            if fid:
                hist = oa_odds_history(fid, market_id=4, bookmaker_id=1)
                oa_open = hist.get("opening", {}) or {}
                oa_now  = hist.get("current", {}) or hist.get("latest", {}) or {}
                oa_peak = hist.get("peak", {}) or {}
                st.write("🟣 OddAlerts (Pinnacle 1X2):")
                st.dataframe(pd.DataFrame([
                    {"Kaynak":"OPEN","H":oa_open.get("home"),"D":oa_open.get("draw"),"A":oa_open.get("away"),
                     "ImpH":implied(oa_open.get("home")),"ImpD":implied(oa_open.get("draw")),"ImpA":implied(oa_open.get("away"))},
                    {"Kaynak":"NOW","H":oa_now.get("home"),"D":oa_now.get("draw"),"A":oa_now.get("away"),
                     "ImpH":implied(oa_now.get("home")),"ImpD":implied(oa_now.get("draw")),"ImpA":implied(oa_now.get("away"))},
                    {"Kaynak":"PEAK","H":oa_peak.get("home"),"D":oa_peak.get("draw"),"A":oa_peak.get("away"),
                     "ImpH":implied(oa_peak.get("home")),"ImpD":implied(oa_peak.get("draw")),"ImpA":implied(oa_peak.get("away"))},
                ]))
            else:
                st.info("OddAlerts: Fixture bulunamadı.")

        # ----- Prediction & Guardrails -----
        st.subheader("📐 Dixon–Coles Tahmin + Guardrails")
        if not MODEL_OK:
            st.error(f"Model modülü yok: {MODEL_ERR}")
        else:
            res = predict_match(
                api_get, home_id, away_id, (league_for_odds or 0), season,
                last5_home_for=hf['GF'], last5_home_against=hf['GA'],
                last5_away_for=af['GF'], last5_away_against=af['GA'],
                odds_map=odds_map, alpha=alpha, max_goals=max_goals
            )
            lam_h = res["lambda_home"]; lam_a = res["lambda_away"]
            p_model = res["probs_dc"]; blended = res["probs_final"]
            labels = {"H": home_team, "D": "Beraberlik", "A": away_team}
            pH, pD, pA = blended["H"], blended["D"], blended["A"]
            winner_key = max(blended, key=blended.get)
            final_human = labels[winner_key]

            st.write(f"λ_home ≈ {lam_h:.2f}, λ_away ≈ {lam_a:.2f}")
            st.write(f"DC Model (H/D/A): {p_model['H']:.2f} / {p_model['D']:.2f} / {p_model['A']:.2f}")
            st.success(f"Olasılıklar (H/D/A): {pH:.2f} / {pD:.2f} / {pA:.2f}")
            st.markdown(f"## 🔮 İlk Seçim: {final_human}")

            # --- Gol olasılıkları (Poisson) ---
            p_home_goal, p_away_goal, p_btts, p_over_05 = calc_goal_probs(lam_h, lam_a)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ev gol atar (≥1)", f"{p_home_goal:.2f}")
            c2.metric("Dep gol atar (≥1)", f"{p_away_goal:.2f}")
            c3.metric("BTTS (Yes)",       f"{p_btts:.2f}")
            c4.metric("Over 0.5",         f"{p_over_05:.2f}")

            # Draw-Guard: otomatik pazar önerisi
            if pD >= draw_guard and winner_key == "H":
                if switch_choice == "1X Double Chance":
                    fair_1x = 1.0 / max(pH + pD, 1e-6)
                    st.info(f"🛡️ Draw-Guard aktif → **1X (Double Chance)** önerisi | Model-fair ≈ {fair_1x:.2f}")
                elif switch_choice == "Home DNB (0)":
                    fair_dnb = (1.0 - pD) / max(pH, 1e-6)
                    st.info(f"🛡️ Draw-Guard aktif → **Home DNB (0)** önerisi | Model-fair ≈ {fair_dnb:.2f}")

            # Min confidence
            if max(pH, pD, pA) < min_conf:
                st.warning(f"Güven düşük (< {min_conf:.2f}). **PAS** uygun olabilir.")

            # CLV guard (OddAlerts'tan)
            if feat_oddalerts and (OA_TOKEN or default_oa):
                side = {"H":"home","D":"draw","A":"away"}[winner_key]
                open_odd = locals().get("oa_open", {}).get(side)
                now_odd  = locals().get("oa_now", {}).get(side)
                if open_odd and now_odd:
                    pct = (now_odd - open_odd) / open_odd * 100.0
                    if pct < clv_guard:
                        st.warning(f"📉 CLV Guard: Seçim oranı açılıştan {pct:.1f}% düştü (aleyhimize akım). **PAS** önerilir.")
                    else:
                        st.caption(f"CLV durumu OK: {pct:.1f}%")

            # -------- Excel export --------
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                summary_df = pd.DataFrame({
                    "Takım":        [home_team,                       away_team],
                    "Son5 G-B-M":   [f"{hf['G']}-{hf['B']}-{hf['M']}", f"{af['G']}-{af['B']}-{af['M']}"],
                    "Son5 GF/GA":   [f"{hf['GF']}/{hf['GA']}",         f"{af['GF']}/{af['GA']}"],
                })
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

                model_df = pd.DataFrame({
                    "Olasılık_H":[pH], "Olasılık_D":[pD], "Olasılık_A":[pA],
                    "lam_home":[lam_h], "lam_away":[lam_a],
                    "P_HomeGoal":[p_home_goal], "P_AwayGoal":[p_away_goal],
                    "P_BTTS":[p_btts], "P_Over0_5":[p_over_05],
                    "IlkSecim":[final_human]
                })
                model_df.to_excel(writer, sheet_name="Summary", index=False, startrow=summary_df.shape[0] + 2)

                if odds_map: pd.DataFrame([odds_map]).to_excel(writer, sheet_name="CurrentOdds", index=False)
                if 'open_map' in locals() and open_map: pd.DataFrame([open_map]).to_excel(writer, sheet_name="OpeningOdds", index=False)
                if (OA_TOKEN or default_oa) and ('oa_open' in locals() or 'oa_now' in locals() or 'oa_peak' in locals()):
                    pd.DataFrame([{
                        "OPEN_H":locals().get("oa_open", {}).get("home"),"OPEN_D":locals().get("oa_open", {}).get("draw"),"OPEN_A":locals().get("oa_open", {}).get("away"),
                        "NOW_H":locals().get("oa_now", {}).get("home"), "NOW_D":locals().get("oa_now", {}).get("draw"), "NOW_A":locals().get("oa_now", {}).get("away"),
                        "PEAK_H":locals().get("oa_peak", {}).get("home"),"PEAK_D":locals().get("oa_peak", {}).get("draw"),"PEAK_A":locals().get("oa_peak", {}).get("away")
                    }]).to_excel(writer, sheet_name="OddAlerts", index=False)

            st.download_button("Excel'i İndir (Tek Maç)", data=buf.getvalue(),
                               file_name=f"master_single_v5c_{home_team}_vs_{away_team}_{season}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------- Batch sekmesi mevcut dosyadan sadeleştirildi (aynı mantık) --------
# İstersen burada da çalıştırırız; cloud için ana senaryo tek maçtır.
