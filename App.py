import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.stats import poisson
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --- 1. UI & TOTAL LEAGUE ARCHITECTURE ---
st.set_page_config(page_title="Pro-Scout v42.0 Total Tier", layout="wide")

TEAM_COLORS = {
    'Galatasaray': ('#890e10', '#fdb912'), 'Fenerbahce': ('#002e5d', '#fbda17'), 
    'Besiktas': ('#000000', '#ffffff'), 'Trabzonspor': ('#800000', '#00a1e1'),
    'Real Madrid': ('#ffffff', '#0047a0'), 'Barcelona': ('#a50044', '#004d98'),
    'Man City': ('#6caddf', '#1c2c5b'), 'Arsenal': ('#ef0107', '#061922'),
    'Liverpool': ('#c8102e', '#f6eb61'), 'Bayern Munich': ('#dc052d', '#0066b2')
}

# TÜRKİYE VE TÜM DÜNYA ALT LİGLERİ (MAKSİMUM KAPSAM)
LIGLER = {
    '🇹🇷 Türkiye (Süper Lig)': 'T1',
    '🇹🇷 Türkiye (1. Lig)': 'T1', # Not: Veri kaynağı bazen T1 içinde alt ligleri de barındırır
    '🏴󠁧󠁢󠁥󠁮󠁧󠁿 İngiltere (Premier Lig)': 'E0',
    '🏴󠁧󠁢󠁥󠁮󠁧󠁿 İngiltere (Championship)': 'E1',
    '🏴󠁧󠁢󠁥󠁮󠁧󠁿 İngiltere (League 1)': 'E2',
    '🏴󠁧󠁢󠁥󠁮󠁧󠁿 İngiltere (League 2)': 'E3',
    '🏴󠁧󠁢󠁥󠁮󠁧󠁿 İngiltere (National League)': 'EC',
    '🇪🇸 İspanya (La Liga)': 'SP1',
    '🇪🇸 İspanya (Segunda)': 'SP2',
    '🇩🇪 Almanya (Bundesliga 1)': 'D1',
    '🇩🇪 Almanya (Bundesliga 2)': 'D2',
    '🇮🇹 İtalya (Serie A)': 'I1',
    '🇮🇹 İtalya (Serie B)': 'I2',
    '🇫🇷 Fransa (Ligue 1)': 'F1',
    '🇫🇷 Fransa (Ligue 2)': 'F2',
    '🇳🇱 Hollanda (Eredivisie)': 'N1',
    '🇧🇪 Belçika (Jupiler Pro)': 'B1',
    '🇵🇹 Portekiz (Primeira Liga)': 'P1',
    '🏴󠁧󠁢󠁳󠁣󠁴󠁿 İskoçya (Premiership)': 'SC0',
    '🏴󠁧󠁢󠁳󠁣󠁴󠁿 İskoçya (Championship)': 'SC1',
    '🏴󠁧󠁢󠁳󠁣󠁴󠁿 İskoçya (League 1)': 'SC2',
    '🏴󠁧󠁢󠁳󠁣󠁴󠁿 İskoçya (League 2)': 'SC3',
    '🇬🇷 Yunanistan (Süper Lig)': 'G1',
    '🇦🇹 Avusturya (Bundesliga)': 'AUT',
    '🇩🇰 Danimarka (Superliga)': 'DNK',
    '🇨🇭 İsviçre (Super League)': 'SWZ'
}

# --- 2. ENGINE & DATA (3-YEAR DEEP DIVE) ---
@st.cache_data
def load_v42(lig_kodu):
    mega = pd.DataFrame()
    current_teams = []
    # 2023'ten 2026'ya kadar tam veri entegrasyonu
    for s in ["2324", "2425", "2526"]:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{s}/{lig_kodu}.csv"
            s_df = pd.read_csv(url)
            mega = pd.concat([mega, s_df], ignore_index=True)
            current_teams = sorted(pd.concat([s_df['HomeTeam'], s_df['AwayTeam']]).unique().tolist())
        except: continue
    
    if mega.empty: return None
    
    # Veri Temizliği ve Eğitim
    df = mega[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG']].dropna()
    le = LabelEncoder().fit(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    df['Ev_K'], df['Dep_K'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    X = df[['Ev_K', 'Dep_K']].values
    
    m_ev, m_dep = RandomForestRegressor(n_estimators=100).fit(X, df['FTHG']), RandomForestRegressor(n_estimators=100).fit(X, df['FTAG'])
    m_ht_ev, m_ht_dep = RandomForestRegressor(n_estimators=100).fit(X, df['HTHG']), RandomForestRegressor(n_estimators=100).fit(X, df['HTAG'])
    
    return mega, m_ev, m_dep, m_ht_ev, m_ht_dep, le, current_teams

def engine_v42(ev_b, dep_b, h_ev_b, h_dep_b, ev_f, dep_f):
    ev_b *= (1 + (ev_f - 0.5) * 0.3); dep_b *= (1 + (dep_f - 0.5) * 0.3)
    ev_p, dep_p = [poisson.pmf(i, ev_b) for i in range(6)], [poisson.pmf(j, dep_b) for j in range(6)]
    m_ft = np.outer(ev_p, dep_p)
    ev_w, ber, dep_w = np.sum(np.tril(m_ft, -1))*100, np.sum(np.diag(m_ft))*100, np.sum(np.triu(m_ft, 1))*100
    u15, u25, kg = sum(m_ft[i,j] for i in range(6) for j in range(6) if i+j > 1.5)*100, sum(m_ft[i,j] for i in range(6) for j in range(6) if i+j > 2.5)*100, (1 - poisson.pmf(0, ev_b)) * (1 - poisson.pmf(0, dep_b)) * 100
    m_ht = np.outer([poisson.pmf(i, h_ev_b) for i in range(4)], [poisson.pmf(j, h_dep_b) for j in range(4)])
    ht_r = {"1": np.sum(np.tril(m_ht, -1)), "0": np.sum(np.diag(m_ht)), "2": np.sum(np.triu(m_ht, 1))}
    ft_r = {"1": ev_w/100, "0": ber/100, "2": dep_w/100}
    htft = sorted([(f"{h}/{f}", ht_r[h]*ft_r[f]*100) for h in ['1','0','2'] for f in ['1','0','2']], key=lambda x: x[1], reverse=True)
    full_skr = sorted([((i,j), m_ft[i,j]*100) for i in range(5) for j in range(5)], key=lambda x: x[1], reverse=True)
    return ev_w, ber, dep_w, u15, u25, kg, htft, full_skr[:8], full_skr[10:13]

# --- 3. UI ASSEMBLY ---
st.markdown("<h2 style='text-align:center;'>🏆 PRO-SCOUT GLOBAL v42.0</h2>", unsafe_allow_html=True)

lig_adi = st.selectbox("🌍 ANALİZ EDİLECEK LİG (ALT LİGLER DAHİL)", list(LIGLER.keys()))
res_v42 = load_v42(LIGLER[lig_adi])

if res_v42:
    mega, m_ev, m_dep, m_ht_ev, m_ht_dep, le, takimlar = res_v42
    c1, c2 = st.columns(2)
    ev_t, dep_t = c1.selectbox("🏠 EV SAHİBİ", takimlar), c2.selectbox("🚀 DEPLASMAN", takimlar)

    if st.button("📊 DERİN ANALİZİ BAŞLAT", use_container_width=True):
        p1, p2 = TEAM_COLORS.get(ev_t, ("#1e293b", "#3b82f6"))
        ev_f = (sum([3 if (r['HomeTeam']==ev_t and r['FTHG']>r['FTAG']) or (r['AwayTeam']==ev_t and r['FTAG']>r['FTHG']) else 1 if r['FTHG']==r['FTAG'] else 0 for _, r in mega[(mega['HomeTeam']==ev_t) | (mega['AwayTeam']==ev_t)].tail(5).iterrows()])/15)
        dep_f = (sum([3 if (r['HomeTeam']==dep_t and r['FTHG']>r['FTAG']) or (r['AwayTeam']==dep_t and r['FTAG']>r['FTHG']) else 1 if mega.loc[_,'FTHG']==mega.loc[_,'FTAG'] else 0 for _, r in mega[(mega['HomeTeam']==dep_t) | (mega['AwayTeam']==dep_t)].tail(5).iterrows()])/15)
        
        # UI: HEADER
        st.markdown(f"<div style='background:linear-gradient(135deg, {p1} 0%, {p2} 100%); color:white; padding:30px; border-radius:15px; text-align:center;'><h1>{ev_t} vs {dep_t}</h1><p>TOTAL TIER v42.0 | GLOBAL DATA</p></div>", unsafe_allow_html=True)

        col_m1, col_m2 = st.columns([1, 1.2])
        with col_m1:
            st.markdown(f"<div style='background:white; padding:20px; border-radius:15px; border-left:10px solid {p1}; text-align:center;'>🎯 <b>SKOR TAHMİNİ</b><br><h1 style='font-size:50px;'>{int(np.round(m_ev.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0]))}-{int(np.round(m_dep.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0]))}</h1></div>", unsafe_allow_html=True)
            evw, ber, depw, u15, u25, kg, htft, top8, surp = engine_v42(m_ev.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0], m_dep.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0], m_ht_ev.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0], m_ht_dep.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0], ev_f, dep_f)
            
            st.markdown("### 📊 EN OLASI 8 SKOR")
            sk_c1, sk_c2 = st.columns(2)
            for i, (sk, pr) in enumerate(top8):
                (sk_c1 if i < 4 else sk_c2).markdown(f"<div style='background:#f1f5f9; padding:8px; border-radius:8px; margin:2px; text-align:center;'><b>{sk[0]}-{sk[1]}</b> (%{pr:.1f})</div>", unsafe_allow_html=True)

        with col_m2:
            st.markdown(f"<div style='background:white; padding:20px; border-radius:15px; border-left:10px solid {p2};'><h4>⚖️ OLASILIK ANALİZİ</h4>", unsafe_allow_html=True)
            st.progress(evw/100, text=f"{ev_t}: %{evw:.1f}"); st.progress(ber/100, text=f"🤝 Beraberlik: %{ber:.1f}"); st.progress(depw/100, text=f"{dep_t}: %{depw:.1f}")
            st.markdown(f"""<div style='background:#f8fafc; padding:15px; border-radius:10px; border:1px solid #ddd; margin-top:10px; text-align:center;'>
            <b>📈 1.5 ÜST: %{u15:.1f}</b> | <b>🔥 2.5 ÜST: %{u25:.1f}</b><br>
            <b>⚽ KG VAR: %{kg:.1f}</b></div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("🔮 9'LU HT/FT TAM MATRİS")
        ht_cols = st.columns(3)
        for i, (res, prob) in enumerate(htft):
            ht_cols[i%3].markdown(f"<div style='background:#f1f5f9; padding:10px; border-radius:8px; text-align:center; border:2.2px solid {p1}; margin-bottom:8px;'><b>{res}</b> (%{prob:.1f})</div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("💡 6 KATMANLI STRATEJİ DANIŞMANI")
        s_c1, s_c2 = st.columns(2)
        s_c1.info(f"💎 **ULTRA-KASA:** {'1.5 ÜST (%{:.0f})'.format(u15) if u15 > 82 else 'Çifte Şans 1X' if evw+ber > 85 else 'Maçın Başını İzle'}")
        s_c1.success(f"🟢 **GÜVENLİ:** {'Karşılıklı Gol (%{:.0f})'.format(kg) if kg > 55 else 'Ev Gol Atar'}")
        s_c1.warning(f"🟡 **ANA TERCİH:** {'2.5 ÜST (%{:.0f})'.format(u25) if u25 > 62 else 'Beraberlikte İade 1'}")
        s_c2.warning(f"🟠 **DEĞERLİ ORAN:** {htft[0][0]} Senaryosu (%{htft[0][1]:.1f})")
        s_c2.error(f"🔴 **BOMBACI (SÜRPRİZ):** {surp[0][0][0]}-{surp[0][0][1]} Skoru (%{surp[0][1]:.1f})")
        s_c2.error(f"🔵 **CANLI AKIŞ:** {'IY 0.5 ÜST' if u15 > 78 else '75. Dakikadan Sonra Aksiyon'}")
