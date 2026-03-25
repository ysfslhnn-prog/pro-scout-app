import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.stats import poisson
import matplotlib.pyplot as plt
from io import BytesIO
import warnings

warnings.filterwarnings('ignore')

# --- 1. PREMIUM UI & ALL TIERS LEAGUES ---
st.set_page_config(page_title="Pro-Scout v48.0 Banko Scanner", layout="wide")

TEAM_COLORS = {
    'Galatasaray': ('#890e10', '#fdb912'), 'Fenerbahce': ('#002e5d', '#fbda17'), 
    'Besiktas': ('#000000', '#ffffff'), 'Trabzonspor': ('#800000', '#00a1e1')
}

LIGLER = {
    '🇹🇷 Türkiye (Süper Lig)': 'T1', '🇹🇷 Türkiye (1. Lig)': 'T2',
    '🏴󠁧󠁢󠁥󠁮󠁧󠁿 İngiltere (Premier)': 'E0', '🏴󠁧󠁢󠁥󠁮󠁧󠁿 İngiltere (Championship)': 'E1',
    '🏴󠁧󠁢󠁥󠁮󠁧󠁿 İngiltere (League 1)': 'E2', '🏴󠁧󠁢󠁥󠁮󠁧󠁿 İngiltere (League 2)': 'E3',
    '🇪🇸 İspanya (La Liga)': 'SP1', '🇪🇸 İspanya (Segunda)': 'SP2',
    '🇩🇪 Almanya (Bundesliga 1)': 'D1', '🇩🇪 Almanya (Bundesliga 2)': 'D2',
    '🇮🇹 İtalya (Serie A)': 'I1', '🇮🇹 İtalya (Serie B)': 'I2',
    '🇫🇷 Fransa (Ligue 1)': 'F1', '🇫🇷 Fransa (Ligue 2)': 'F2',
    '🇳🇱 Hollanda (Eredivisie)': 'N1', '🇧🇪 Belçika (Jupiler Pro)': 'B1',
    '🇵🇹 Portekiz (Primeira Liga)': 'P1', '🏴󠁧󠁢󠁳󠁣󠁴󠁿 İskoçya (Premiership)': 'SC0',
    '🏴󠁧󠁢󠁳󠁣󠁴󠁿 İskoçya (Championship)': 'SC1', '🇬🇷 Yunanistan (Süper Lig)': 'G1'
}

# --- 2. ENGINE & ANALYSIS CORE ---
@st.cache_data
def master_load_v48(lig_kodu):
    mega = pd.DataFrame()
    active_2026 = []
    for s in ["2324", "2425", "2526"]:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{s}/{lig_kodu}.csv"
            s_df = pd.read_csv(url); mega = pd.concat([mega, s_df], ignore_index=True)
            if s == "2526": active_2026 = sorted(pd.concat([s_df['HomeTeam'], s_df['AwayTeam']]).unique().tolist())
        except: continue
    if mega.empty: return None
    if not active_2026: active_2026 = sorted(pd.concat([mega['HomeTeam'], mega['AwayTeam']]).unique().tolist())
    df = mega[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG']].dropna()
    le = LabelEncoder().fit(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    df['Ev_K'], df['Dep_K'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    X = df[['Ev_K', 'Dep_K']].values
    m_ev, m_dep = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['FTHG']), RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['FTAG'])
    m_ht_ev, m_ht_dep = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['HTHG']), RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['HTAG'])
    return mega, m_ev, m_dep, m_ht_ev, m_ht_dep, le, active_2026

def engine_v48(ev_b, dep_b, h_ev_b, h_dep_b, ev_f, dep_f):
    ev_b *= (1 + (ev_f - 0.5) * 0.3); dep_b *= (1 + (dep_f - 0.5) * 0.3)
    ev_p, dep_p = [poisson.pmf(i, ev_b) for i in range(6)], [poisson.pmf(j, dep_b) for j in range(6)]
    m_ft = np.outer(ev_p, dep_p)
    ev_w, ber, dep_w = np.sum(np.tril(m_ft, -1))*100, np.sum(np.diag(m_ft))*100, np.sum(np.triu(m_ft, 1))*100
    u15, u25, kg = sum(m_ft[i,j] for i in range(6) for j in range(6) if i+j > 1.5)*100, sum(m_ft[i,j] for i in range(6) for j in range(6) if i+j > 2.5)*100, (1 - poisson.pmf(0, ev_b)) * (1 - poisson.pmf(0, dep_b)) * 100
    m_ht = np.outer([poisson.pmf(i, h_ev_b) for i in range(4)], [poisson.pmf(j, h_dep_b) for j in range(4)])
    ht_r = {"1": np.sum(np.tril(m_ht, -1)), "0": np.sum(np.diag(m_ht)), "2": np.sum(np.triu(m_ht, 1))}
    htft = sorted([(f"{h}/{f}", ht_r[h]*( (ev_w/100) if f=='1' else (ber/100) if f=='0' else (dep_w/100))*100) for h in ['1','0','2'] for f in ['1','0','2']], key=lambda x: x[1], reverse=True)
    skr = sorted([((i,j), m_ft[i,j]*100) for i in range(5) for j in range(5)], key=lambda x: x[1], reverse=True)
    return ev_w, ber, dep_w, u15, u25, kg, htft, skr[:8], skr[10:13]

def rekabet_v48(df, ev, dep):
    h2h = df[((df['HomeTeam'] == ev) & (df['AwayTeam'] == dep)) | ((df['HomeTeam'] == dep) & (df['AwayTeam'] == ev))]
    if h2h.empty: return {"status": False, "msg": "Kayıt yok."}
    ew = len(h2h[(h2h['HomeTeam'] == ev) & (h2h['FTHG'] > h2h['FTAG'])]) + len(h2h[(h2h['AwayTeam'] == ev) & (h2h['FTAG'] > h2h['FTHG'])])
    dw = len(h2h[(h2h['HomeTeam'] == dep) & (h2h['FTHG'] > h2h['FTAG'])]) + len(h2h[(h2h['AwayTeam'] == dep) & (h2h['FTAG'] > h2h['FTHG'])])
    dr = len(h2h[h2h['FTHG'] == h2h['FTAG']])
    return {"status": True, "total": len(h2h), "ev_w": ew, "dep_w": dw, "draw": dr, "avg": round((h2h['FTHG'].sum() + h2h['FTAG'].sum())/len(h2h), 2)}

# --- 3. UI DASHBOARD ---
tab_ana, tab_banko = st.tabs(["📊 DERİN ANALİZ", "🔍 GÜNÜN BANKOLARI"])

with tab_ana:
    lig_sel = st.selectbox("🌍 LİG SEÇİN", list(LIGLER.keys()))
    res = master_load_v48(LIGLER[lig_sel])
    if res:
        mega, m_ev, m_dep, m_ht_ev, m_ht_dep, le, active_teams = res
        c1, c2 = st.columns(2)
        ev_t, dep_t = c1.selectbox("🏠 EV", active_teams), c2.selectbox("🚀 DEP", active_teams)

        if st.button("📊 ANALİZİ BAŞLAT", use_container_width=True):
            p1, p2 = TEAM_COLORS.get(ev_t, ("#1e293b", "#3b82f6"))
            ev_f = (sum([3 if (r['HomeTeam']==ev_t and r['FTHG']>r['FTAG']) or (r['AwayTeam']==ev_t and r['FTAG']>r['FTHG']) else 1 if r['FTHG']==r['FTAG'] else 0 for _, r in mega[(mega['HomeTeam']==ev_t) | (mega['AwayTeam']==ev_t)].tail(5).iterrows()])/15)
            dep_f = (sum([3 if (r['HomeTeam']==dep_t and r['FTHG']>r['FTAG']) or (r['AwayTeam']==dep_t and r['FTAG']>r['FTHG']) else 1 if mega.loc[_,'FTHG']==mega.loc[_,'FTAG'] else 0 for _, r in mega[(mega['HomeTeam']==dep_t) | (mega['AwayTeam']==dep_t)].tail(5).iterrows()])/15)
            fe, fd = m_ev.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0], m_dep.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0]
            he, hd = m_ht_ev.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0], m_ht_dep.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0]
            evw, ber, depw, u15, u25, kg, htft, top8, surp = engine_v48(fe, fd, he, hd, ev_f, dep_f)
            h2h = rekabet_v48(mega, ev_t, dep_t)

            st.markdown(f"<div style='background:linear-gradient(90deg, {p1} 0%, {p2} 100%); color:white; padding:25px; border-radius:15px; text-align:center;'><h2>{ev_t} - {dep_t}</h2></div>", unsafe_allow_html=True)
            
            cm1, cm2 = st.columns([1, 1.2])
            with cm1:
                st.markdown(f"<div style='background:white; padding:20px; border-radius:15px; border-left:10px solid {p1}; text-align:center;'><b>🎯 SKOR TAHMİNİ</b><br><h1 style='margin:0;'>{int(np.round(fe))}-{int(np.round(fd))}</h1><small>İY: {int(np.round(he))}-{int(np.round(hd))}</small></div>", unsafe_allow_html=True)
                st.write("### 📊 EN OLASI 8 SKOR")
                sk_c1, sk_c2 = st.columns(2)
                for i, (sk, pr) in enumerate(top8):
                    (sk_c1 if i < 4 else sk_c2).markdown(f"<div style='background:#f1f5f9; padding:5px; border-radius:5px; margin:2px; text-align:center;'><b>{sk[0]}-{sk[1]}</b> (%{pr:.1f})</div>", unsafe_allow_html=True)
            with cm2:
                st.write("### ⚖️ OLASILIKLAR")
                st.progress(evw/100, text=f"{ev_t}: %{evw:.1f} {'⭐' if evw > 80 else ''}")
                st.progress(ber/100, text=f"🤝 Beraberlik: %{ber:.1f}")
                st.progress(depw/100, text=f"{dep_t}: %{depw:.1f}")
                st.markdown(f"<div style='background:#f8fafc; padding:15px; border-radius:10px; border:1px solid #ddd; margin-top:10px; text-align:center;'><b>📈 1.5 ÜST: %{u15:.1f}</b> | <b>🔥 2.5 ÜST: %{u25:.1f}</b><br><b>⚽ KG VAR: %{kg:.1f}</b></div>", unsafe_allow_html=True)
                if h2h['status']: st.info(f"⚔️ **REKABET:** {h2h['ev_w']} - {h2h['draw']} - {h2h['dep_w']} | Ort. {h2h['avg']} Gol")

            st.divider()
            st.subheader("🔮 9'LU HT/FT MATRİSİ")
            ht_cols = st.columns(3)
            for i, (res, prob) in enumerate(htft):
                ht_cols[i%3].markdown(f"<div style='background:#f1f5f9; padding:10px; border-radius:8px; text-align:center; border:2px solid {p1}; margin-bottom:8px;'><b>{res}</b><br>%{prob:.1f}</div>", unsafe_allow_html=True)

            st.divider()
            st.subheader("💡 6 KATMANLI STRATEJİ")
            s_c1, s_c2 = st.columns(2)
            s_c1.info(f"💎 **ULTRA-KASA:** {'1.5 ÜST' if u15 > 80 else 'Çifte Şans 1X' if evw+ber > 82 else 'Pas'}")
            s_c1.success(f"🟢 **GÜVENLİ:** {'Karşılıklı Gol' if kg > 60 else 'Maç Sonucu 1' if evw > 62 else 'Ev Gol Atar'}")
            s_c1.warning(f"🟡 **ANA TERCİH:** {'2.5 ÜST' if u25 > 62 else 'İY 0.5 ÜST'}")
            s_c2.warning(f"🟠 **DEĞERLİ ORAN:** {htft[0][0]} Senaryosu")
            s_c2.error(f"🔴 **BOMBACI:** {surp[0][0][0]}-{surp[0][0][1]} Skoru")
            s_c2.error(f"🔵 **CANLI:** {'75. Dakika Gol' if u15 > 75 else 'Kart Bahsi'}")

with tab_banko:
    st.subheader("🔍 TÜM LİGLERDE 0.8+ BANKO TARAMASI")
    if st.button("🚀 GÜNÜN BANKOLARINI LİSTELE", use_container_width=True):
        with st.spinner("Tüm ligler taranıyor..."):
            found = []
            for name, code in LIGLER.items():
                data_b = master_load_v48(code)
                if data_b:
                    m_mega, m_ev_b, m_dep_b, m_ht_ev_b, m_ht_dep_b, m_le, m_teams = data_b
                    # Son 2-3 maçlık veriye göre örnek tarama (Performans için örneklem)
                    # Not: Gerçek bülten taraması için maç fikstürü gerekir, burada olasılık motoru test edilir.
                    for i in range(min(5, len(m_teams)-1)):
                        t1, t2 = m_teams[i], m_teams[i+1]
                        fe, fd = m_ev_b.predict([[m_le.transform([t1])[0], m_le.transform([t2])[0]]])[0], m_dep_b.predict([[m_le.transform([t1])[0], m_le.transform([t2])[0]]])[0]
                        ew, b, dw, u15, u25, kg, hf, tk, sp = engine_v48(fe, fd, 0, 0, 0.5, 0.5)
                        if any(p > 82 for p in [ew, u15, u25]):
                            found.append({"Lig": name, "Maç": f"{t1} - {t2}", "Banko": "1.5 ÜST" if u15 > 82 else "MS 1" if ew > 82 else "2.5 ÜST", "Güven": f"%{max(ew, u15, u25):.1f}"})
            if found:
                st.table(found)
            else:
                st.write("Şu an kriterlere uygun maç bulunamadı.")
