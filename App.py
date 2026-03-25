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

# --- 1. AYARLAR ---
st.set_page_config(page_title="Pro-Scout Precision", layout="wide")

TAKIM_RENKLERI = {
    'Galatasaray': ('#fdb912', '#890e10'), 'Fenerbahce': ('#002e5d', '#fbda17'), 
    'Besiktas': ('#000000', '#ffffff'), 'Trabzonspor': ('#800000', '#00a1e1'),
    'Real Madrid': ('#ffffff', '#3e3181'), 'Barcelona': ('#a50044', '#004d98'),
    'Man City': ('#6caddf', '#1c2c5b'), 'Arsenal': ('#ef0107', '#061922'),
    'Liverpool': ('#c8102e', '#00b2a9'), 'Bayern Munich': ('#dc052d', '#0066b2')
}

LIGLER = {'Türkiye (Süper Lig)': 'T1', 'İngiltere (Premier Lig)': 'E0', 'İspanya (La Liga)': 'SP1', 'Almanya (Bundesliga)': 'D1', 'İtalya (Serie A)': 'I1', 'Fransa (Ligue 1)': 'F1', 'Hollanda (Eredivisie)': 'N1'}

# --- 2. GÜNCEL VERİ MOTORU ---
@st.cache_data
def load_and_filter_v31(lig_kodu):
    mega = pd.DataFrame()
    current_season_teams = []
    # Son 3 sezonu çek ama sadece en güncel olanın takımlarını filtrele
    seasons = ["2324", "2425", "2526"]
    for s in seasons:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{s}/{lig_kodu}.csv"
            season_df = pd.read_csv(url)
            mega = pd.concat([mega, season_df], ignore_index=True)
            # Liste her zaman en son başarılı yüklenen sezonun takımlarını tutar
            current_season_teams = sorted(pd.concat([season_df['HomeTeam'], season_df['AwayTeam']]).unique().tolist())
        except: continue
    
    if mega.empty: return None
    
    df = mega[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG']].dropna()
    le = LabelEncoder().fit(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    df['Ev_K'], df['Dep_K'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    X = df[['Ev_K', 'Dep_K']].values
    
    m_ev, m_dep = RandomForestRegressor(n_estimators=100).fit(X, df['FTHG']), RandomForestRegressor(n_estimators=100).fit(X, df['FTAG'])
    m_ht_ev, m_ht_dep = RandomForestRegressor(n_estimators=100).fit(X, df['HTHG']), RandomForestRegressor(n_estimators=100).fit(X, df['HTAG'])
    
    return mega, m_ev, m_dep, m_ht_ev, m_ht_dep, le, current_season_teams

def engine_v31(ev_b, dep_b, h_ev_b, h_dep_b, ev_f, dep_f):
    ev_b *= (1 + (ev_f - 0.5) * 0.3); dep_b *= (1 + (dep_f - 0.5) * 0.3)
    ev_p, dep_p = [poisson.pmf(i, ev_b) for i in range(6)], [poisson.pmf(j, dep_b) for j in range(6)]
    m_ft = np.outer(ev_p, dep_p)
    ev_w, ber, dep_w = np.sum(np.tril(m_ft, -1))*100, np.sum(np.diag(m_ft))*100, np.sum(np.triu(m_ft, 1))*100
    u15, u25, kg = sum(m_ft[i,j] for i in range(6) for j in range(6) if i+j > 1.5)*100, sum(m_ft[i,j] for i in range(6) for j in range(6) if i+j > 2.5)*100, (1 - poisson.pmf(0, ev_b)) * (1 - poisson.pmf(0, dep_b)) * 100
    m_ht = np.outer([poisson.pmf(i, h_ev_b) for i in range(4)], [poisson.pmf(j, h_dep_b) for j in range(4)])
    ht_r = {"1": np.sum(np.tril(m_ht, -1)), "0": np.sum(np.diag(m_ht)), "2": np.sum(np.triu(m_ht, 1))}
    ft_r = {"1": ev_w/100, "0": ber/100, "2": dep_w/100}
    htft = sorted([(f"{h}/{f}", ht_r[h]*ft_r[f]*100) for h in ['1','0','2'] for f in ['1','0','2']], key=lambda x: x[1], reverse=True)
    skr = sorted([((i,j), m_ft[i,j]*100) for i in range(5) for j in range(5)], key=lambda x: x[1], reverse=True)
    return ev_w, ber, dep_w, u15, u25, kg, htft, skr[:3], skr[4:7]

# --- 3. INFOGRAPHIC ---
def get_report_img(ev, dep, ms, iy, htft, kg, u25, p1, p2):
    fig, ax = plt.subplots(figsize=(6, 8), facecolor=p1)
    ax.axis('off')
    plt.text(0.5, 0.9, "PRO-SCOUT REPORT", color='white', fontsize=20, ha='center', weight='bold')
    plt.text(0.5, 0.82, f"{ev} vs {dep}", color='white', fontsize=14, ha='center', bbox=dict(facecolor=p2, alpha=0.8))
    content = f"SKOR: {ms}\nİY: {iy}\nHT/FT: {htft}\nKG VAR: %{kg:.1f}\n2.5 ÜST: %{u25:.1f}"
    plt.text(0.5, 0.45, content, color='black', fontsize=16, ha='center', va='center', bbox=dict(facecolor='white', boxstyle='round,pad=1.2', edgecolor=p2, lw=3))
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=120)
    return buf.getvalue()

# --- 4. UI ---
st.title("🏆 PRO-SCOUT PRECISION v31.0")
lig_box = st.selectbox("🌍 LİG SEÇİN", list(LIGLER.keys()))
data_res = load_and_filter_v31(LIGLER[lig_box])

if data_res:
    mega_df, m_ev, m_dep, m_ht_ev, m_ht_dep, le, active_teams = data_res
    c1, c2 = st.columns(2)
    ev_t = c1.selectbox("🏠 EV SAHİBİ (GÜNCEL)", active_teams)
    dep_t = c2.selectbox("🚀 DEPLASMAN (GÜNCEL)", active_teams)

    if st.button("📊 ANALİZİ BAŞLAT", use_container_width=True):
        p1, p2 = TAKIM_RENKLERI.get(ev_t, ("#1e293b", "#3b82f6"))
        ev_f = (sum([3 if (r['HomeTeam']==ev_t and r['FTHG']>r['FTAG']) or (r['AwayTeam']==ev_t and r['FTAG']>r['FTHG']) else 1 if r['FTHG']==r['FTAG'] else 0 for _, r in mega_df[(mega_df['HomeTeam']==ev_t) | (mega_df['AwayTeam']==ev_t)].tail(5).iterrows()])/15)
        dep_f = (sum([3 if (r['HomeTeam']==dep_t and r['FTHG']>r['FTAG']) or (r['AwayTeam']==dep_t and r['FTAG']>r['FTHG']) else 1 if mega_df.loc[_,'FTHG']==mega_df.loc[_,'FTAG'] else 0 for _, r in mega_df[(mega_df['HomeTeam']==dep_t) | (mega_df['AwayTeam']==dep_t)].tail(5).iterrows()])/15)
        
        g = [[le.transform([ev_t])[0], le.transform([dep_t])[0]]]
        fe, fd, he, hd = m_ev.predict(g)[0], m_dep.predict(g)[0], m_ht_ev.predict(g)[0], m_ht_dep.predict(g)[0]
        evw, ber, depw, u15, u25, kg, htft, top3, surp = engine_v31(fe, fd, he, hd, ev_f, dep_f)

        st.markdown(f"<div style='background:{p1}; color:white; padding:15px; border-radius:10px; text-align:center; border-bottom:5px solid {p2}'><h2>{ev_t} - {dep_t}</h2></div>", unsafe_allow_html=True)
        
        col_r1, col_r2 = st.columns(2)
        col_r1.metric("SKOR TAHMİNİ", f"{int(np.round(fe))}-{int(np.round(fd))}", f"İY: {int(np.round(he))}-{int(np.round(hd))}")
        col_r2.write(f"**OLASILIKLAR:**\n\n🏠 %{evw:.1f} | 🤝 %{ber:.1f} | 🚀 %{depw:.1f}")

        st.subheader("🔮 9'LU HT/FT TAM MATRİS")
        h_cols = st.columns(3)
        for i, (res, prob) in enumerate(htft):
            h_cols[i%3].markdown(f"<div style='background:#f1f5f9; padding:10px; border-radius:5px; text-align:center; border:1px solid {p1}; margin-bottom:5px;'><b>{res}</b><br>%{prob:.1f}</div>", unsafe_allow_html=True)
        
        st.divider()
        st.write(f"✅ **2.5 ÜST:** %{u25:.1f} | ⚽ **KG VAR:** %{kg:.1f} | 🚀 **FORM:** %{ev_f*100:.0f} vs %{dep_f*100:.0f}")
        st.error(f"💣 **SÜRPRİZ SKOR:** {surp[0][0][0]}-{surp[0][0][1]} (%{surp[0][1]:.1f})")
        
        report = get_report_img(ev_t, dep_t, f"{int(np.round(fe))}-{int(np.round(fd))}", f"{int(np.round(he))}-{int(np.round(hd))}", htft[0][0], kg, u25, p1, p2)
        st.download_button("🖼️ KUPON GÖRSELİNİ İNDİR", data=report, file_name="analiz.png", mime="image/png", use_container_width=True)
