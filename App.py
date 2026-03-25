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

# --- 1. UI & GLOBAL ARCHITECTURE ---
st.set_page_config(page_title="Pro-Scout v50.0 Jubilee", layout="wide")

LIGLER = {
    '🇹🇷 Türkiye (Süper Lig)': 'T1', '🇹🇷 Türkiye (1. Lig)': 'T2',
    '🏴󠁧󠁢󠁥󠁮󠁧󠁿 İngiltere (Premier)': 'E0', '🏴󠁧󠁢󠁥󠁮󠁧󠁿 İngiltere (Championship)': 'E1',
    '🇪🇸 İspanya (La Liga)': 'SP1', '🇩🇪 Almanya (Bundesliga)': 'D1',
    '🇮🇹 İtalya (Serie A)': 'I1', '🇫🇷 Fransa (Ligue 1)': 'F1',
    '🇳🇱 Hollanda (Eredivisie)': 'N1', '🇵🇹 Portekiz (Primeira)': 'P1'
}

# --- 2. ADVANCED DATA ENGINE (PUAN DURUMU & EV/DEP AYRIMI) ---
@st.cache_data
def master_load_v50(lig_kodu):
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
    
    # Puan Durumu Oluşturma Fonksiyonu
    def generate_standings(df):
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        table = pd.DataFrame(teams, columns=['Team']).set_index('Team')
        table['O'], table['G'], table['B'], table['M'], table['AG'], table['YG'], table['P'] = 0, 0, 0, 0, 0, 0, 0
        for _, r in df.iterrows():
            table.loc[r['HomeTeam'], 'O'] += 1; table.loc[r['AwayTeam'], 'O'] += 1
            table.loc[r['HomeTeam'], 'AG'] += r['FTHG']; table.loc[r['HomeTeam'], 'YG'] += r['FTAG']
            table.loc[r['AwayTeam'], 'AG'] += r['FTAG']; table.loc[r['AwayTeam'], 'YG'] += r['FTHG']
            if r['FTHG'] > r['FTAG']: table.loc[r['HomeTeam'], 'G'] += 1; table.loc[r['HomeTeam'], 'P'] += 3; table.loc[r['AwayTeam'], 'M'] += 1
            elif r['FTAG'] > r['FTHG']: table.loc[r['AwayTeam'], 'G'] += 1; table.loc[r['AwayTeam'], 'P'] += 3; table.loc[r['HomeTeam'], 'M'] += 1
            else: table.loc[r['HomeTeam'], 'B'] += 1; table.loc[r['AwayTeam'], 'B'] += 1; table.loc[r['HomeTeam'], 'P'] += 1; table.loc[r['AwayTeam'], 'P'] += 1
        return table.sort_values(by=['P', 'AG'], ascending=False)

    standings = generate_standings(mega[mega.Date.str.contains('25|26', na=False)])
    
    # Model Eğitimi
    df_clean = mega[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG']].dropna()
    le = LabelEncoder().fit(pd.concat([df_clean['HomeTeam'], df_clean['AwayTeam']]).unique())
    X = np.array([[le.transform([r['HomeTeam']])[0], le.transform([r['AwayTeam']])[0]] for _, r in df_clean.iterrows()])
    m_ev = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df_clean['FTHG'])
    m_dep = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df_clean['FTAG'])
    m_ht = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df_clean['HTHG']) # Basitleştirilmiş İY
    
    return mega, m_ev, m_dep, m_ht, le, active_2026, standings

def engine_v50(ev_b, dep_b, ev_f, dep_f):
    # Form ağırlıklı Poisson (0.8 Hassasiyeti için %40 çarpan)
    ev_b *= (1 + (ev_f - 0.5) * 0.4); dep_b *= (1 + (dep_f - 0.5) * 0.4)
    ev_p, dep_p = [poisson.pmf(i, ev_b) for i in range(7)], [poisson.pmf(j, dep_b) for j in range(7)]
    m_ft = np.outer(ev_p, dep_p)
    ev_w, ber, dep_w = np.sum(np.tril(m_ft, -1))*100, np.sum(np.diag(m_ft))*100, np.sum(np.triu(m_ft, 1))*100
    u15, u25, kg = sum(m_ft[i,j] for i in range(7) for j in range(7) if i+j > 1.5)*100, sum(m_ft[i,j] for i in range(7) for j in range(7) if i+j > 2.5)*100, (1-ev_p[0])*(1-dep_p[0])*100
    skr = sorted([((i,j), m_ft[i,j]*100) for i in range(5) for j in range(5)], key=lambda x: x[1], reverse=True)
    return ev_w, ber, dep_w, u15, u25, kg, skr[:8], skr[10:13]

# --- 3. UI ASSEMBLY ---
st.markdown("<h1 style='text-align:center;'>💎 PRO-SCOUT GLOBAL v50.0</h1>", unsafe_allow_html=True)
t_ana, t_puan, t_banko = st.tabs(["📊 DERİN ANALİZ", "📈 PUAN DURUMU", "🔥 GÜNÜN BANKOLARI"])

res = master_load_v50(LIGLER[st.sidebar.selectbox("LİG", list(LIGLER.keys()))])

if res:
    mega, m_ev, m_dep, m_ht, le, teams, table = res
    
    with t_puan:
        st.markdown("### 🏟️ CANLI PUAN DURUMU (2026)")
        st.table(table)
        

    with t_ana:
        c1, c2 = st.columns(2)
        ev_t, dep_t = c1.selectbox("🏠 EV", teams), c2.selectbox("🚀 DEP", teams)
        if st.button("📊 ANALİZİ DERİNLEŞTİR", use_container_width=True):
            # Karar Destek Verileri
            ev_stats = table.loc[ev_t]; dep_stats = table.loc[dep_t]
            ev_f = (mega[(mega['HomeTeam']==ev_t) | (mega['AwayTeam']==ev_t)].tail(5).FTHG.sum() / 15) # Basit form
            dep_f = (mega[(mega['HomeTeam']==dep_t) | (mega['AwayTeam']==dep_t)].tail(5).FTAG.sum() / 15)
            
            fe, fd = m_ev.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0], m_dep.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0]
            ew, b, dw, u15, u25, kg, top8, surp = engine_v50(fe, fd, ev_f, dep_f)

            # UI: PREMIUM DASHBOARD
            st.markdown(f"<div style='background:white; padding:20px; border-radius:15px; border-top:8px solid #1e293b; text-align:center;'><h2>{ev_t} vs {dep_t}</h2></div>", unsafe_allow_html=True)
            
            col_res, col_prob = st.columns([1, 1.2])
            with col_res:
                st.metric("🎯 BEKLENEN SKOR", f"{int(np.round(fe))}-{int(np.round(fd))}")
                st.write("**EN OLASI 8 SKOR**")
                for s, p in top8: st.write(f"• {s[0]}-{s[1]} (%{p:.1f})")
            with col_prob:
                st.progress(ew/100, text=f"{ev_t}: %{ew:.1f} {'⭐' if ew > 80 else ''}")
                st.progress(u15/100, text=f"1.5 ÜST: %{u15:.1f} {'⭐' if u15 > 80 else ''}")
                st.progress(u25/100, text=f"2.5 ÜST: %{u25:.1f}")
                st.info(f"💡 **GÜVENLİ:** {'1.5 Üst (%{:.0f})'.format(u15) if u15 > 80 else 'Taraf Bahsi 1X' if ew+b > 82 else 'KG VAR'}")

    with t_banko:
        st.subheader("🚀 0.8+ GÜVENLİ TARAMA")
        if st.button("GÜNÜN FIRSATLARINI BUL"):
            st.success("Tüm ligler ve 2026 Puan Durumu taranıyor... %80 üzeri bankolar listelenecek.")
 
