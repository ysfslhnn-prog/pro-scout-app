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

# --- 1. PREMIUM UI & THEME ---
st.set_page_config(page_title="Pro-Scout Masterpiece", layout="wide")

TAKIM_RENKLERI = {
    'Galatasaray': ('#fdb912', '#890e10'), 'Fenerbahce': ('#002e5d', '#fbda17'), 
    'Besiktas': ('#000000', '#ffffff'), 'Trabzonspor': ('#800000', '#00a1e1'),
    'Real Madrid': ('#ffffff', '#3e3181'), 'Barcelona': ('#a50044', '#004d98'),
    'Man City': ('#6caddf', '#1c2c5b'), 'Arsenal': ('#ef0107', '#061922'),
    'Liverpool': ('#c8102e', '#00b2a9'), 'Bayern Munich': ('#dc052d', '#0066b2')
}

LIGLER = {'Türkiye (Süper Lig)': 'T1', 'İngiltere (Premier Lig)': 'E0', 'İspanya (La Liga)': 'SP1', 'Almanya (Bundesliga)': 'D1', 'İtalya (Serie A)': 'I1', 'Fransa (Ligue 1)': 'F1', 'Hollanda (Eredivisie)': 'N1'}

# --- 2. ENGINE & DATA (PRECISION 2026) ---
@st.cache_data
def load_and_filter_v32(lig_kodu):
    mega = pd.DataFrame()
    current_teams = []
    for s in ["2324", "2425", "2526"]:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{s}/{lig_kodu}.csv"
            s_df = pd.read_csv(url)
            mega = pd.concat([mega, s_df], ignore_index=True)
            current_teams = sorted(pd.concat([s_df['HomeTeam'], s_df['AwayTeam']]).unique().tolist())
        except: continue
    if mega.empty: return None
    df = mega[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG']].dropna()
    le = LabelEncoder().fit(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    df['Ev_K'], df['Dep_K'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    X = df[['Ev_K', 'Dep_K']].values
    m_ev, m_dep = RandomForestRegressor(n_estimators=100).fit(X, df['FTHG']), RandomForestRegressor(n_estimators=100).fit(X, df['FTAG'])
    m_ht_ev, m_ht_dep = RandomForestRegressor(n_estimators=100).fit(X, df['HTHG']), RandomForestRegressor(n_estimators=100).fit(X, df['HTAG'])
    return mega, m_ev, m_dep, m_ht_ev, m_ht_dep, le, current_teams

def engine_v32(ev_b, dep_b, h_ev_b, h_dep_b, ev_f, dep_f):
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
def draw_v32_card(ev, dep, ms, iy, htft, kg, u25, p1, p2):
    fig, ax = plt.subplots(figsize=(6, 8.5), facecolor=p1)
    ax.axis('off')
    plt.text(0.5, 0.92, "PRO-SCOUT MASTER REPORT", color='white', fontsize=22, ha='center', weight='bold')
    plt.text(0.5, 0.84, f"{ev} - {dep}", color='white', fontsize=16, ha='center', bbox=dict(facecolor=p2, alpha=0.9))
    content = f"SKOR: {ms}\nİY: {iy}\nHT/FT: {htft}\nKG VAR: %{kg:.1f}\n2.5 ÜST: %{u25:.1f}"
    plt.text(0.5, 0.45, content, color='black', fontsize=18, ha='center', va='center', bbox=dict(facecolor='white', boxstyle='round,pad=1', edgecolor=p2, lw=4))
    buf = BytesIO(); plt.savefig(buf, format="png", bbox_inches='tight', dpi=130); return buf.getvalue()

# --- 4. DASHBOARD ASSEMBLY ---
st.markdown("<h1 style='text-align:center;'>🏆 PRO-SCOUT v32.0</h1>", unsafe_allow_html=True)

lig_box = st.selectbox("🌍 LİG SEÇİN", list(LIGLER.keys()))
res = load_and_filter_v32(LIGLER[lig_box])

if res:
    mega_df, m_ev, m_dep, m_ht_ev, m_ht_dep, le, active_teams = res
    c1, c2 = st.columns(2)
    ev_t = c1.selectbox("🏠 EV SAHİBİ", active_teams)
    dep_t = c2.selectbox("🚀 DEPLASMAN", active_teams)

    if st.button("📊 DERİN ANALİZİ BAŞLAT", use_container_width=True):
        p1, p2 = TAKIM_RENKLERI.get(ev_t, ("#1e293b", "#3b82f6"))
        ev_f = (sum([3 if (r['HomeTeam']==ev_t and r['FTHG']>r['FTAG']) or (r['AwayTeam']==ev_t and r['FTAG']>r['FTHG']) else 1 if r['FTHG']==r['FTAG'] else 0 for _, r in mega_df[(mega_df['HomeTeam']==ev_t) | (mega_df['AwayTeam']==ev_t)].tail(5).iterrows()])/15)
        dep_f = (sum([3 if (r['HomeTeam']==dep_t and r['FTHG']>r['FTAG']) or (r['AwayTeam']==dep_t and r['FTAG']>r['FTHG']) else 1 if mega_df.loc[_,'FTHG']==mega_df.loc[_,'FTAG'] else 0 for _, r in mega_df[(mega_df['HomeTeam']==dep_t) | (mega_df['AwayTeam']==dep_t)].tail(5).iterrows()])/15)
        
        g = [[le.transform([ev_t])[0], le.transform([dep_t])[0]]]
        fe, fd, he, hd = m_ev.predict(g)[0], m_dep.predict(g)[0], m_ht_ev.predict(g)[0], m_ht_dep.predict(g)[0]
        evw, ber, depw, u15, u25, kg, htft, top3, surp = engine_v32(fe, fd, he, hd, ev_f, dep_f)

        # UI KARTLARI
        st.markdown(f"""<div style="background:{p1}; color:white; padding:15px; border-radius:12px; text-align:center; border-bottom:6px solid {p2}; margin-bottom:20px;">
        <h2 style="margin:0">{ev_t} vs {dep_t}</h2>
        <small>Form Momentum: %{ev_f*100:.0f} vs %{dep_f*100:.0f}</small></div>""", unsafe_allow_html=True)

        col_main1, col_main2 = st.columns(2)
        with col_main1:
            st.metric("TABELA TAHMİNİ", f"{int(np.round(fe))}-{int(np.round(fd))}", f"İY: {int(np.round(he))}-{int(np.round(hd))}")
            st.markdown(f"**OLASI SKORLAR:**\n\n• {top3[0][0][0]}-{top3[0][0][1]} (%{top3[0][1]:.1f})\n• {top3[1][0][0]}-{top3[1][0][1]} (%{top3[1][1]:.1f})")
        
        with col_main2:
            st.write("**OLASILIKLAR**")
            st.progress(evw/100, text=f"{ev_t}: %{evw:.1f}")
            st.progress(ber/100, text=f"Beraberlik: %{ber:.1f}")
            st.progress(depw/100, text=f"{dep_t}: %{depw:.1f}")
            st.markdown(f"<div style='background:#f8fafc; padding:10px; border-radius:8px; border:1px solid #ddd;'><b>GOL ANALİZİ:</b><br>2.5 ÜST: %{u25:.1f} | 1.5 ÜST: %{u15:.1f}<br><b>KG VAR: %{kg:.1f}</b></div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("🔮 9'LU HT/FT MATRİS ANALİZİ")
        ht_cols = st.columns(3)
        for i, (res, prob) in enumerate(htft):
            ht_cols[i%3].markdown(f"<div style='background:#f1f5f9; padding:10px; border-radius:8px; text-align:center; border:1.5px solid {p1}; margin-bottom:10px;'><b>{res}</b><br><small>%{prob:.1f}</small></div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("💡 STRATEJİ DANIŞMANI")
        st.info(f"🟢 **GÜVENLİ:** {'1.5 Üst (%{:.0f})'.format(u15) if u15 > 75 else 'Çifte Şans 1X' if evw+ber > 80 else 'Maçın gidişatını gör.'}")
        st.warning(f"🟡 **ANA TERCİH:** {'2.5 Üst (%{:.0f})'.format(u25) if u25 > 62 else 'KG VAR (%{:.0f})'.format(kg) if kg > 58 else 'Taraf Bahsi (MS)'}")
        st.error(f"💣 **SÜRPRİZ SKOR:** {surp[0][0][0]}-{surp[0][0][1]} (%{surp[0][1]:.1f})")
        
        report = draw_v32_card(ev_t, dep_t, f"{int(np.round(fe))}-{int(np.round(fd))}", f"{int(np.round(he))}-{int(np.round(hd))}", htft[0][0], kg, u25, p1, p2)
        st.download_button("🖼️ KUPON GÖRSELİNİ İNDİR", data=report, file_name=f"{ev_t}_scout.png", mime="image/png", use_container_width=True)
