import os
import math
import time
import json
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ---------------------------
# CONFIG
# ---------------------------
APP_TITLE = "⚽ Football Master — ECO v2 (Low API, Smart Thresholds)"
API_BASE  = "https://v3.football.api-sports.io"
DEFAULT_TTL = 6 * 60 * 60  # 6 hours cache TTL

st.set_page_config(page_title=f"{APP_TITLE}", layout="wide")
st.title(APP_TITLE)
st.caption("Minimum API, maksimum netlik. Oyuncu/lineup yok, sadece maç-temelli + oran (opsiyonel) + akıllı eşiklerle öneri.")

# ---------------------------
# KEYS & OPTIONS
# ---------------------------
with st.sidebar:
    st.header("🔐 Anahtarlar")
    API_KEY = st.text_input("API-Football PRO Key", type="password", value=os.environ.get("APIFOOTBALL_KEY", ""))
    if not API_KEY:
        st.warning("API anahtarı boş. Env veya burada gir.")
        st.stop()

    st.header("⚙️ Seçenekler")
    # Akıllı sezon varsayılanı: Temmuzdan sonra = o yıl
    now = datetime.now()
    default_season = now.year if now.month >= 7 else now.year - 1
    season = st.number_input("Sezon (başlangıç yılı)", min_value=2015, max_value=2026, value=default_season, step=1)

    use_odds_api  = st.checkbox("Oranları çek (fixture-id varsa)", True)
    allow_league_fallback = st.checkbox("Fixture bulunamazsa lig+sezon fallback (ekstra 1 istek)", False)
    allow_manual_odds = st.checkbox("Manuel 1X2 odds gir", True)

    st.header("⚖️ Analiz Eşikleri")
    btts_h_thr = st.slider("BTTS — p(Ev≥1) eşiği", 0.40, 0.80, 0.60, 0.01)
    btts_a_thr = st.slider("BTTS — p(Dep≥1) eşiği", 0.40, 0.80, 0.55, 0.01)
    o15_lambda_thr = st.slider("Over 1.5 — λ_total eşiği", 1.8, 3.2, 2.20, 0.05)
    o25_lambda_thr = st.slider("Over 2.5 — λ_total eşiği", 2.2, 3.8, 2.80, 0.05)

    st.divider()
    st.caption("ECO çağrılar: leagues (1) • teams (1) • fixtures home (1) • fixtures away (1) • odds fixture (1, opsiyonel).")

# ---------------------------
# SESSION STATE CACHE & STATS
# ---------------------------
if "_mem_cache" not in st.session_state:
    st.session_state["_mem_cache"] = {}  # key -> {"ts": epoch, "data": obj}
if "_api_stats" not in st.session_state:
    st.session_state["_api_stats"] = {"calls_made": 0, "hits_saved": 0, "keys": set()}

def _cache_key(endpoint: str, params: dict|None):
    if params is None:
        return (endpoint, None)
    items = tuple(sorted((str(k), str(v)) for k,v in params.items()))
    return (endpoint, items)

def api_get(endpoint: str, params: dict|None=None, ttl: int=DEFAULT_TTL):
    """Minimalistic memoized GET with TTL and call stats."""
    headers = {"x-apisports-key": API_KEY}
    key = _cache_key(endpoint, params or {})
    now = time.time()

    cache = st.session_state["_mem_cache"]
    stats = st.session_state["_api_stats"]

    # cache hit?
    rec = cache.get(key)
    if rec and (now - rec["ts"] <= ttl):
        stats["hits_saved"] += 1
        return rec["data"]

    # miss -> request
    url = f"{API_BASE}/{endpoint}"
    try:
        r = requests.get(url, headers=headers, params=params, timeout=25)
        stats["calls_made"] += 1
        stats["keys"].add(f"{endpoint}|{json.dumps(params, ensure_ascii=False)}")
        if r.status_code != 200:
            st.error(f"API Hatası {r.status_code} — {endpoint} {params}\n{r.text[:500]}")
            return {}
        js = r.json()
        cache[key] = {"ts": now, "data": js}
        return js
    except Exception as e:
        st.error(f"API çağrısı hata: {e}")
        return {}

def to_utc(dt_str: str) -> datetime:
    if not dt_str:
        return datetime.now(timezone.utc)
    s = dt_str.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        try:
            dt = datetime.fromisoformat(s.split("T")[0] + "T00:00:00+00:00")
        except Exception:
            return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt

# ---------------------------
# LIGHTWEIGHT HELPERS
# ---------------------------
def leagues_list():
    data = api_get("leagues")
    out = [{"id": it["league"]["id"], "name": it["league"]["name"], "country": it["country"]["name"]}
           for it in data.get("response", [])]
    return pd.DataFrame(out)

def teams_in(league_id, season):
    data = api_get("teams", {"league": league_id, "season": season})
    out = [{"id": t["team"]["id"], "name": t["team"]["name"]} for t in data.get("response", [])]
    return pd.DataFrame(out)

def fixtures_by_team(team_id, season):
    # Tek istek: tüm sezon. Sonra biz kısıyoruz.
    data = api_get("fixtures", {"team": team_id, "season": season})
    return data.get("response", [])

def lastN(team_id, N=6):
    """Sadece tek çağrıya dayanır (fixtures_by_team)."""
    fx = fixtures_by_team(team_id, season)
    finished = [f for f in fx if f["fixture"]["status"]["short"] in ("FT","AET","PEN")]
    finished.sort(key=lambda x: to_utc(x["fixture"]["date"]), reverse=True)
    sel = finished[:N]

    rows = []
    w=d=l=0; gf=ga=0; cnt=0
    home_home_gf=[]; away_away_gf=[]; home_home_ga=[]; away_away_ga=[]
    for f in sel:
        hg = f["score"]["fulltime"]["home"]; ag = f["score"]["fulltime"]["away"]
        if hg is None or ag is None: continue
        cnt += 1
        is_home = (f["teams"]["home"]["id"] == team_id)
        if is_home:
            gf += hg; ga += ag
            home_home_gf.append(hg); home_home_ga.append(ag)
        else:
            gf += ag; ga += hg
            away_away_gf.append(ag); away_away_ga.append(hg)

        if (hg>ag and is_home) or (ag>hg and not is_home): w+=1
        elif hg==ag: d+=1
        else: l+=1
        rows.append({
            "Tarih": f["fixture"]["date"],
            "Ev": f["teams"]["home"]["name"],
            "Dep": f["teams"]["away"]["name"],
            "Skor": f"{hg}-{ag}",
            "Durum": f["fixture"]["status"]["short"]
        })
    stat = {
        "G": w, "B": d, "M": l,
        "GF_avg": round(gf/max(cnt,1), 2),
        "GA_avg": round(ga/max(cnt,1), 2),
        "H_GF_home_avg": np.mean(home_home_gf) if home_home_gf else np.nan,
        "H_GA_home_avg": np.mean(home_home_ga) if home_home_ga else np.nan,
        "A_GF_away_avg": np.mean(away_away_gf) if away_away_gf else np.nan,
        "A_GA_away_avg": np.mean(away_away_ga) if away_away_ga else np.nan,
        "count": cnt
    }
    return stat, pd.DataFrame(rows)

def simple_poisson_lambdas(h_stat, a_stat):
    """ECO model: evde ev hücumu + deplasmanda deplasmanın hücumu, buna karşı rakip savunmaları birleşimi (yumuşak)."""
    # Hücum göstergeleri
    h_att = h_stat.get("H_GF_home_avg") if not np.isnan(h_stat.get("H_GF_home_avg")) else h_stat.get("GF_avg", 1.2)
    a_att = a_stat.get("A_GF_away_avg") if not np.isnan(a_stat.get("A_GF_away_avg")) else a_stat.get("GF_avg", 1.0)
    # Savunma göstergeleri
    h_def = a_stat.get("A_GA_away_avg") if not np.isnan(a_stat.get("A_GA_away_avg")) else a_stat.get("GA_avg", 1.0)
    a_def = h_stat.get("H_GA_home_avg") if not np.isnan(h_stat.get("H_GA_home_avg")) else h_stat.get("GA_avg", 1.2)
    # Harman
    lam_home = 0.6 * h_att + 0.4 * h_def * 0.95  # ev avantajı küçük buff
    lam_away = 0.6 * a_att + 0.4 * a_def
    # Güvenlik sınırları
    lam_home = float(np.clip(lam_home, 0.3, 3.5))
    lam_away = float(np.clip(lam_away, 0.2, 3.2))
    return lam_home, lam_away

def poisson_probs(lam_h, lam_a, max_goals=6):
    """Basit 1X2 olasılıkları: iki bağımsız Poisson ve skor matrisi."""
    i = np.arange(0, max_goals+1, dtype=int)
    fac = np.vectorize(math.factorial)(i)
    ph = np.exp(-lam_h) * np.power(lam_h, i) / fac
    pa = np.exp(-lam_a) * np.power(lam_a, i) / fac
    mat = np.outer(ph, pa)
    pH = np.tril(mat, -1).sum()    # home > away
    pD = np.trace(mat)
    pA = np.triu(mat,  1).sum()
    return {"H": float(pH), "D": float(pD), "A": float(pA)}

def goal_side_probs(lam_h, lam_a):
    """Pratik olasılıklar: takım gol atar mı, BTTS, toplam gol >0.5"""
    p_home_goal = 1.0 - math.exp(-lam_h)
    p_away_goal = 1.0 - math.exp(-lam_a)
    p_btts = 1.0 - math.exp(-lam_h) - math.exp(-lam_a) + math.exp(-(lam_h + lam_a))
    p_over_05 = 1.0 - math.exp(-(lam_h + lam_a))
    return p_home_goal, p_away_goal, p_btts, p_over_05

def calc_ou(lam_h, lam_a):
    lam_t = lam_h + lam_a
    # P(Over 1.5) = 1 - P(0) - P(1) for Poisson(lam_t)
    p0 = math.exp(-lam_t) * (lam_t**0) / math.factorial(0)
    p1 = math.exp(-lam_t) * (lam_t**1) / math.factorial(1)
    over15 = 1.0 - (p0 + p1)
    # P(Over 2.5) = 1 - sum_{k=0..2} P(k)
    p2 = math.exp(-lam_t) * (lam_t**2) / math.factorial(2)
    over25 = 1.0 - (p0 + p1 + p2)
    return {"Over1.5": over15, "Over2.5": over25, "Lambda_Total": lam_t}

def get_fixture_id(home_id, away_id, season):
    fx = fixtures_by_team(home_id, season)
    for f in fx:
        if f["teams"]["home"]["id"] == home_id and f["teams"]["away"]["id"] == away_id:
            return f["fixture"]["id"], f
    return None, None

def fetch_odds_fixture(fixture_id):
    data = api_get("odds", {"fixture": fixture_id})
    rows = []
    for it in data.get("response", []):
        for bm in it.get("bookmakers", []) or []:
            for bet in bm.get("bets", []) or []:
                nm = (bet.get("name") or "").strip().lower()
                if nm in ("match winner", "1x2", "1×2", "1 x 2"):
                    for val in bet.get("values", []) or []:
                        rows.append({"Bookmaker": bm.get("name"), "Sonuç": val.get("value"), "Oran": float(val.get("odd"))})
    return pd.DataFrame(rows)

def fetch_odds_league(league_id, season, home_id, away_id):
    """Opsiyonel fallback: 1 ek istek. Lig+sezon odds'tan ilgili eşleşmeyi bulmaya çalışır."""
    data = api_get("odds", {"league": league_id, "season": season})
    rows = []
    for it in data.get("response", []):
        try:
            ih = it["teams"]["home"]["id"]; ia = it["teams"]["away"]["id"]
        except Exception:
            continue
        if not (ih == home_id and ia == away_id):
            continue
        for bm in it.get("bookmakers", []) or []:
            for bet in bm.get("bets", []) or []:
                nm = (bet.get("name") or "").strip().lower()
                if nm in ("match winner", "1x2", "1×2", "1 x 2"):
                    for val in bet.get("values", []) or []:
                        rows.append({"Bookmaker": bm.get("name"), "Sonuç": val.get("value"), "Oran": float(val.get("odd"))})
    return pd.DataFrame(rows)

# ---------------------------
# UI: Team selection
# ---------------------------
st.subheader("Takım Seçimi (ECO v2)")
leagues_df = leagues_list()
if leagues_df.empty:
    st.error("Lig verisi alınamadı.")
    st.stop()

sel_labels = leagues_df.apply(lambda r: f"{r['country']} - {r['name']}", axis=1).tolist()
sel_league_name = st.selectbox("Lig", sel_labels)
sel_league_id = int(leagues_df.iloc[sel_labels.index(sel_league_name)]["id"])

teams_df = teams_in(sel_league_id, season)
if teams_df.empty:
    st.error("Takım verisi alınamadı.")
    st.stop()

col1, col2 = st.columns(2)
home_team = col1.selectbox("Ev Sahibi", teams_df["name"].tolist())
away_team = col2.selectbox("Deplasman", teams_df["name"].tolist(), index=1 if len(teams_df)>1 else 0)

try:
    home_id = int(teams_df.loc[teams_df["name"]==home_team, "id"].iloc[0])
    away_id = int(teams_df.loc[teams_df["name"]==away_team, "id"].iloc[0])
except Exception:
    st.error("Takım ID tespit edilemedi.")
    st.stop()

# ---------------------------
# ANALYZE
# ---------------------------
if st.button("Analiz (ECO v2)"):
    # LastN stats (2 çağrı: team fixtures)
    h_stat, h_df = lastN(home_id, N=6)
    a_stat, a_df = lastN(away_id, N=6)

    c1, c2 = st.columns(2)
    c1.metric(f"{home_team} G-B-M", f"{h_stat['G']}-{h_stat['B']}-{h_stat['M']}")
    c1.metric(f"{home_team} GF/GA (avg)", f"{h_stat['GF_avg']}/{h_stat['GA_avg']}")
    c2.metric(f"{away_team} G-B-M", f"{a_stat['G']}-{a_stat['B']}-{a_stat['M']}")
    c2.metric(f"{away_team} GF/GA (avg)", f"{a_stat['GF_avg']}/{a_stat['GA_avg']}")

    with st.expander(f"{home_team} • Son 6"):
        st.dataframe(h_df if not h_df.empty else pd.DataFrame([{"📄": "Veri yok"}]))
    with st.expander(f"{away_team} • Son 6"):
        st.dataframe(a_df if not a_df.empty else pd.DataFrame([{"📄": "Veri yok"}]))

    # Simple model (0 ekstra çağrı)
    lam_h, lam_a = simple_poisson_lambdas(h_stat, a_stat)
    probs = poisson_probs(lam_h, lam_a, max_goals=6)
    st.subheader("📐 ECO Poisson Model (1X2)")
    st.write(f"λ_home ≈ {lam_h:.2f}, λ_away ≈ {lam_a:.2f}")
    st.success(f"Olasılıklar (H/D/A): {probs['H']:.2f} / {probs['D']:.2f} / {probs['A']:.2f}")

    p_home_goal, p_away_goal, p_btts, p_over_05 = goal_side_probs(lam_h, lam_a)
    ou = calc_ou(lam_h, lam_a)

    c3, c4, c5, c6 = st.columns(4)
    c3.metric("Ev gol atar (≥1)", f"{p_home_goal:.2f}")
    c4.metric("Dep gol atar (≥1)", f"{p_away_goal:.2f}")
    c5.metric("BTTS (Yes)",       f"{p_btts:.2f}")
    c6.metric("Over 0.5",         f"{p_over_05:.2f}")

    c7, c8, c9 = st.columns(3)
    c7.metric("Over 1.5",  f"{ou['Over1.5']:.2f}")
    c8.metric("Over 2.5",  f"{ou['Over2.5']:.2f}")
    c9.metric("λ total",   f"{ou['Lambda_Total']:.2f}")

    # ---------------- SMART THRESHOLD SIGNALS ----------------
    st.subheader("🧠 Eşik Bazlı Sinyaller (ECO)")
    btts_ok = (p_home_goal >= btts_h_thr) and (p_away_goal >= btts_a_thr)
    o15_ok  = (ou["Lambda_Total"] >= o15_lambda_thr)
    o25_ok  = (ou["Lambda_Total"] >= o25_lambda_thr)

    s1 = "✅ **BTTS Yes** uygun" if btts_ok else "⚠️ **BTTS Yes** riskli"
    s2 = "✅ **Over 1.5** uygun" if o15_ok else "⚠️ **Over 1.5** riskli"
    s3 = "✅ **Over 2.5** uygun" if o25_ok else "⚠️ **Over 2.5** riskli"

    st.write(f"- {s1}  •  p(Ev≥1)={p_home_goal:.2f}  /  p(Dep≥1)={p_away_goal:.2f}  (eşikler: {btts_h_thr:.2f}, {btts_a_thr:.2f})")
    st.write(f"- {s2}  •  λ_total={ou['Lambda_Total']:.2f} (eşik: {o15_lambda_thr:.2f})")
    st.write(f"- {s3}  •  λ_total={ou['Lambda_Total']:.2f} (eşik: {o25_lambda_thr:.2f})")

    # ---------------- ORANLAR (opsiyonel) ----------------
    odds_map = None
    fixture_id = None
    fx_obj = None
    fixture_id, fx_obj = get_fixture_id(home_id, away_id, season)  # 0 ekstra; home fixtures'tan
    if use_odds_api and fixture_id:
        df_odds = fetch_odds_fixture(fixture_id)  # +1 çağrı
        if df_odds.empty and allow_league_fallback:
            df_odds = fetch_odds_league(sel_league_id, season, home_id, away_id)  # +1 çağrı
        if not df_odds.empty:
            st.subheader("🎰 Oranlar (ECO v2)")
            st.dataframe(df_odds)
            pvt = df_odds.pivot_table(index="Sonuç", values="Oran", aggfunc=np.median).to_dict()["Oran"]
            normalize = {"Home":"Home","1":"Home","home":"Home","Draw":"Draw","X":"Draw","draw":"Draw","Away":"Away","2":"Away","away":"Away"}
            odds_map = {}
            for k,v in pvt.items():
                kk = normalize.get(str(k), None)
                if kk: odds_map[kk] = float(v)
        else:
            st.info("Oran bulunamadı (fixture için). Lig fallback kapalı olabilir.")

    # ---------------- QUICK VALUE CHECK (opsiyonel) ----------------
    st.subheader("💡 Hızlı Value Kontrol (opsiyonel)")
    st.caption("Aşağıya tek bir pazar için oran gir; model olasılığıyla kıyaslayalım.")
    market = st.selectbox("Pazar", ["1X2 — Ev", "1X2 — Beraberlik", "1X2 — Deplasman", "BTTS — Yes", "Over 1.5", "Over 2.5"], index=0)
    odd_in = st.text_input("Desimal oran (örn. 1.80) — boş bırakabilirsin", value="")
    model_p = None
    if market == "1X2 — Ev": model_p = probs["H"]
    elif market == "1X2 — Beraberlik": model_p = probs["D"]
    elif market == "1X2 — Deplasman": model_p = probs["A"]
    elif market == "BTTS — Yes": model_p = p_btts
    elif market == "Over 1.5": model_p = ou["Over1.5"]
    elif market == "Over 2.5": model_p = ou["Over2.5"]

    if odd_in.strip():
        try:
            dec = float(odd_in.replace(",", "."))
            imp = 1.0/dec if dec>0 else None
        except Exception:
            imp = None
        if imp:
            edge = model_p - imp
            verdict = "✅ Value var" if edge > 0.02 else ("⚠️ Sınırda" if -0.01 <= edge <= 0.02 else "❌ Value yok")
            st.write(f"- Model p ≈ **{model_p:.3f}**, Implied ≈ **{imp:.3f}**, Edge ≈ **{edge:.3f}** → {verdict}")

    # ---------------- STATS ----------------
    st.divider()
    stats = st.session_state["_api_stats"]
    st.write("### API Çağrı Özeti (bu oturum)")
    st.write({
        "calls_made": stats["calls_made"],
        "cache_hits_saved": stats["hits_saved"],
        "unique_keys": len(stats["keys"])
    })
    with st.expander("Detay (unique istekler)"):
        st.code("\n".join(sorted(stats["keys"])) or "—", language="text")
