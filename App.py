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

# --- 1. PREMIUM UI & COUNTRY LEAGUES ONLY ---
st.set_page_config(page_title="Pro-Scout v47.0 Pure League", layout="wide")

TEAM_COLORS = {
    'Galatasaray': ('#890e10', '#fdb912'), 'Fenerbahce': ('#002e5d', '#fbda17'), 
    'Besiktas': ('#000000', '#ffffff'), 'Trabzonspor': ('#800000', '#00a1e1'),
    'Real Madrid': ('#ffffff', '#0047a0'), 'Barcelona': ('#a50044', '#004d98'),
    'Man City': ('#6caddf', '#1c2c5b'), 'Arsenal': ('#ef0107', '#061922'),
    'Liverpool': ('#c8102e', '#f6eb61'), 'Bayern Munich': ('#dc052d', '#0066b2')
}

# SADECE ÜLKE LİGLERİ (ALT LİGLER DAHİL)
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
    '🇬🇷 Yunanistan (Süper Lig)': 'G1', '🇦🇹 Avusturya (Bundesliga)': 'AUT',
    '🇩🇰 Danimarka (Superliga)': 'DNK', '🇨🇭 İsviçre (Super League)': 'SWZ'
}

# --- 2. CORE ENGINE: 3-YEAR DEEP DIVE & 2026 FILTER ---
@st.cache_data
def master_load_v47(lig_kodu):
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

def rekabet_pro_v47(df, ev, dep):
    h2h = df[((df['HomeTeam'] == ev) & (df['AwayTeam'] == dep)) | ((df['HomeTeam'] == dep) & (df['AwayTeam'] == ev))]
    if h2h.empty: return {"status": False, "msg": "Rekabet kaydı bulunamadı."}
    ev_w = len(h2h[(h2h['HomeTeam'] == ev) & (h2h['FTHG'] > h2h['FTAG'])]) + len(h2h[(h2h['AwayTeam'] == ev) & (h2h['FTAG'] > h2h['FTHG'])])
    dep_w = len(h2h[(h2h['HomeTeam'] == dep) & (h2h['FTHG'] > h2h['FTAG'])]) + len(h2h[(h2h['AwayTeam'] == dep) & (h2h['FTAG'] > h2h['FTHG'])])
    draw = len(h2h[h2h['FTHG'] == h2h['FTAG']])
    avg = round((h2h['FTHG'].sum() + h2h['FTAG'].sum())/len(h2h), 2)
    ev_cs = len(h2h[(h2h['HomeTeam'] == ev) & (h2h['FTAG'] == 0)]) + len(h2h[(h2h['AwayTeam'] == ev) & (h2h['FTHG'] == 0)])
    return {"status": True, "total": len(h2h), "ev_w": ev_w, "dep_w": dep_w, "draw": draw, "avg": avg, "ev_cs": ev_cs}

def engine_v47(ev_b, dep_b, h_ev_b, h_dep_b, ev_f, dep_f):
    ev_b *= (1 + (ev_f - 0.5) * 0.3); dep_b *= (1 + (dep_f - 0.5) * 0.3)
    ev_p, dep_p = [poisson.pmf(i, ev_b) for i in range(6)], [poisson.pmf(j, dep_b) for j in range(6)]
    m_ft = np.outer(ev_p, dep_p)
    ev_w, ber, dep_w = np.sum(np.tril(m_ft, -1))*100, np.sum(np.diag(m_ft))*100, np.sum(np.triu(m_ft, 1))*100
    u15, u25, kg = sum(m_ft[i,j] for i in range(6) for j in range(6) if i+j > 1.5)*100, sum(m_ft[i,j] for i in range(6) for j in range(6) if i+j > 2.5)*100, (1 - poisson.pmf(0, ev_b)) * (1 - poisson.pmf(0, dep_b)) * 100
    m_ht = np.outer([poisson.pmf(i, h_ev_b) for i in range(4)], [poisson.pmf(j, h_dep_b) for j in range(4)])
    ht_r = {"1": np.sum(np.tril(m_ht, -1)), "0": np.sum(np.diag(m_ht)), "2": np.sum(np.triu(m_ht, 1))}
    htft = sorted([(f"{h}/{f}", ht_r[h]*( (ev_w/100) if f=='1' else (ber/100) if f=='0' else (dep_w/100))*100) for h in ['1','0','2'] for f in ['1','0','2']], key=lambda x: x[1], reverse=True)
    full_skr = sorted([((i,j), m_ft[i,j]*100) for i in range(5) for j in range(5)], key=lambda x: x[1], reverse=True)
    return ev_w, ber, dep_w, u15, u25, kg, htft, full_skr[:8], full_skr[10:13]

# --- 3. UI ASSEMBLY ---
st.markdown("<h2 style='text-align:center;'>🏆 PRO-SCOUT PURE LEAGUE v47.0</h2>", unsafe_allow_html=True)
lig_sel = st.selectbox("🌍 LİG SEÇİN (2026 TAKIMLARI)", list(LIGLER.keys()))
res = master_load_v47(LIGLER[lig_sel])

if res:
    mega, m_ev, m_dep, m_ht_ev, m_ht_dep, le, active_teams = res
    c1, c2 = st.columns(2)
    ev_t, dep_t = c1.selectbox("🏠 EV", active_teams), c2.selectbox("🚀 DEP", active_teams)

    if st.button("📊 DERİN ANALİZİ BAŞLAT", use_container_width=True):
        p1, p2 = TEAM_COLORS.get(ev_t, ("#1e293b", "#3b82f6"))
        ev_f = (sum([3 if (r['HomeTeam']==ev_t and r['FTHG']>r['FTAG']) or (r['AwayTeam']==ev_t and r['FTAG']>r['FTHG']) else 1 if r['FTHG']==r['FTAG'] else 0 for _, r in mega[(mega['HomeTeam']==ev_t) | (mega['AwayTeam']==ev_t)].tail(5).iterrows()])/15)
        dep_f = (sum([3 if (r['HomeTeam']==dep_t and r['FTHG']>r['FTAG']) or (r['AwayTeam']==dep_t and r['FTAG']>r['FTHG']) else 1 if mega.loc[_,'FTHG']==mega.loc[_,'FTAG'] else 0 for _, r in mega[(mega['HomeTeam']==dep_t) | (mega['AwayTeam']==dep_t)].tail(5).iterrows()])/15)
        h2h = rekabet_pro_v47(mega, ev_t, dep_t)
        
        # Olasılıklar
        fe, fd = m_ev.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0], m_dep.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0]
        he, hd = m_ht_ev.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0], m_ht_dep.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0]
        evw, ber, depw, u15, u25, kg, htft, top8, surp = engine_v47(fe, fd, he, hd, ev_f, dep_f)

        # 0.8 Confidence Check
        if any(p > 80 for p in [evw, ber, depw, u15, u25, (evw+ber)]):
            st.balloons()
            st.markdown(f"<div style='background:#fef3c7; color:#92400e; padding:15px; border-radius:10px; text-align:center; font-weight:bold; border:2px solid #f59e0b;'>⭐ PREMIUM TERCİH: 0.8 GÜVEN BARAJI GEÇİLDİ!</div>", unsafe_allow_html=True)

        st.markdown(f"<div style='background:linear-gradient(135deg, {p1} 0%, {p2} 100%); color:white; padding:30px; border-radius:15px; text-align:center;'><h1>{ev_t} - {dep_t}</h1><p>v47.0 | ARCHITECT'S FINAL BLUEPRINT</p></div>", unsafe_allow_html=True)

        col_m1, col_m2 = st.columns([1, 1.2])
        with col_m1:
            st.markdown(f"<div style='background:white; padding:20px; border-radius:15px; border-left:10px solid {p1}; text-align:center;'>🎯 <b>SKOR: {int(np.round(fe))}-{int(np.round(fd))}</b></div>", unsafe_allow_html=True)
            st.markdown("### 📊 EN OLASI 8 SKOR")
            sk_c1, sk_c2 = st.columns(2)
            for i, (sk, pr) in enumerate(top8):
                (sk_c1 if i < 4 else sk_c2).markdown(f"<div style='background:#f1f5f9; padding:8px; border-radius:8px; margin:2px; text-align:center;'><b>{sk[0]}-{sk[1]}</b> (%{pr:.1f})</div>", unsafe_allow_html=True)

        with col_m2:
            st.markdown(f"<div style='background:white; padding:20px; border-radius:15px; border-left:10px solid {p2};'><h4>⚖️ OLASILIK ANALİZİ</h4>", unsafe_allow_html=True)
            st.progress(evw/100, text=f"{ev_t}: %{evw:.1f} {'⭐' if evw > 80 else ''}")
            st.progress(ber/100, text=f"🤝 Beraberlik: %{ber:.1f}")
            st.progress(depw/100, text=f"{dep_t}: %{depw:.1f}")
            st.markdown(f"<div style='background:#f8fafc; padding:15px; border-radius:10px; border:1px solid #ddd; margin-top:10px; text-align:center;'>📈 1.5 ÜST: %{u15:.1f} | 🔥 2.5 ÜST: %{u25:.1f} | ⚽ KG VAR: %{kg:.1f}</div>", unsafe_allow_html=True)
            if h2h['status']: st.info(f"⚔️ **H2H REKABET:** {h2h['ev_w']} - {h2h['draw']} - {h2h['dep_w']} | Ort. {h2h['avg']} Gol | {ev_t} {h2h['ev_cs']} maçta gol yemedi.")

        st.divider()
        st.subheader("🔮 9'LU HT/FT TAM MATRİS")
        ht_cols = st.columns(3)
        for i, (res, prob) in enumerate(htft):
            ht_cols[i%3].markdown(f"<div style='background:#f1f5f9; padding:12px; border-radius:10px; text-align:center; border:2px solid {p1}; margin-bottom:12px;'><b>{res}</b><br>%{prob:.1f}</div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("💡 6 KATMANLI STRATEJİ DANIŞMANI")
        s_c1, s_c2 = st.columns(2)
        s_c1.info(f"💎 **ULTRA-KASA:** {'1.5 ÜST' if u15 > 80 else 'Çifte Şans 1X' if evw+ber > 82 else 'Pas'}")
        s_c1.success(f"🟢 **GÜVENLİ:** {'Maç Sonucu 1' if evw > 62 else 'Karşılıklı Gol' if kg > 60 else 'Ev Gol Atar'}")
        s_c1.warning(f"🟡 **ANA TERCİH:** {'2.5 ÜST' if u25 > 62 else 'Handikaplı 1' if evw > 75 else 'Maç Sonucu 1'}")
        s_c2.warning(f"🟠 **DEĞERLİ ORAN:** {htft[0][0]} Senaryosu")
        s_c2.error(f"🔴 **BOMBACI (SÜRPRİZ):** {surp[0][0][0]}-{surp[0][0][1]} Skoru")
        s_c2.error(f"🔵 **CANLI AKIŞ:** {'IY 0.5 ÜST' if u15 > 78 else '75. Dakika Gol Kokusu'}")
        
        # PNG Downloader
        plt.figure(figsize=(6, 4), facecolor=p1)
        plt.text(0.5, 0.5, f"{ev_t} vs {dep_t}\nSkor: {int(np.round(fe))}-{int(np.round(fd))}\nKG: %{kg:.1f}", ha='center', color='white', weight='bold')
        plt.axis('off')
        buf = BytesIO(); plt.savefig(buf, format="png"); plt.close()
        st.download_button("🖼️ KUPON GÖRSELİNİ İNDİR", data=buf.getvalue(), file_name=f"{ev_t}_analiz.png", mime="image/png", use_container_width=True)
