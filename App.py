import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.stats import poisson
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Pro-Scout Mobile", layout="wide")

# --- 2. DİNAMİK TEMA VE VERİ AYARLARI ---
TAKIM_RENKLERI = {
    'Galatasaray': ('#fdb912', '#890e10'), 'Fenerbahce': ('#002e5d', '#fbda17'), 
    'Besiktas': ('#000000', '#ffffff'), 'Trabzonspor': ('#800000', '#00a1e1'),
    'Real Madrid': ('#ffffff', '#3e3181'), 'Barcelona': ('#a50044', '#004d98'),
    'Man City': ('#6caddf', '#1c2c5b'), 'Arsenal': ('#ef0107', '#061922'),
    'Liverpool': ('#c8102e', '#00b2a9'), 'Bayern Munich': ('#dc052d', '#0066b2')
}

LIGLER = {'Türkiye (Süper Lig)': 'T1', 'İngiltere (Premier Lig)': 'E0', 'İspanya (La Liga)': 'SP1', 'Almanya (Bundesliga)': 'D1', 'İtalya (Serie A)': 'I1', 'Fransa (Ligue 1)': 'F1', 'Hollanda (Eredivisie)': 'N1'}

@st.cache_data
def load_and_train(lig_kodu):
    mega = pd.DataFrame()
    for s in ["2324", "2425", "2526"]:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{s}/{lig_kodu}.csv"
            mega = pd.concat([mega, pd.read_csv(url)], ignore_index=True)
        except: continue
    
    if mega.empty: return None, None, None, None, None, None, []
    
    df = mega[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG']].dropna()
    le = LabelEncoder().fit(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    df['Ev_K'], df['Dep_K'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    X = df[['Ev_K', 'Dep_K']].values
    m_ev = RandomForestRegressor(n_estimators=50).fit(X, df['FTHG'])
    m_dep = RandomForestRegressor(n_estimators=50).fit(X, df['FTAG'])
    m_ht_ev = RandomForestRegressor(n_estimators=50).fit(X, df['HTHG'])
    m_ht_dep = RandomForestRegressor(n_estimators=50).fit(X, df['HTAG'])
    return mega, m_ev, m_dep, m_ht_ev, m_ht_dep, le, sorted(le.classes_)

# --- 3. ANALİZ MOTORU (Full Specs) ---
def engine_v27(ev_b, dep_b, h_ev_b, h_dep_b, ev_f, dep_f):
    ev_b *= (1 + (ev_f - 0.5) * 0.3); dep_b *= (1 + (dep_f - 0.5) * 0.3)
    ev_p, dep_p = [poisson.pmf(i, ev_b) for i in range(6)], [poisson.pmf(j, dep_b) for j in range(6)]
    m_ft = np.outer(ev_p, dep_p)
    ev_w, ber, dep_w = np.sum(np.tril(m_ft, -1))*100, np.sum(np.diag(m_ft))*100, np.sum(np.triu(m_ft, 1))*100
    u25, kg = sum(m_ft[i,j] for i in range(6) for j in range(6) if i+j > 2.5)*100, (1 - poisson.pmf(0, ev_b)) * (1 - poisson.pmf(0, dep_b)) * 100
    m_ht = np.outer([poisson.pmf(i, h_ev_b) for i in range(4)], [poisson.pmf(j, h_dep_b) for j in range(4)])
    ht_r = {"1": np.sum(np.tril(m_ht, -1)), "0": np.sum(np.diag(m_ht)), "2": np.sum(np.triu(m_ht, 1))}
    ft_r = {"1": ev_w/100, "0": ber/100, "2": dep_w/100}
    htft = sorted([(f"{h}/{f}", ht_r[h]*ft_r[f]*100) for h in ['1','0','2'] for f in ['1','0','2']], key=lambda x: x[1], reverse=True)
    skr = sorted([((i,j), m_ft[i,j]*100) for i in range(5) for j in range(5)], key=lambda x: x[1], reverse=True)
    return ev_w, ber, dep_w, u25, kg, htft, skr[:3], skr[4]

# --- 4. MOBİL ARAYÜZ ---
st.title("🏆 PRO-SCOUT MOBILE")
lig_adi = st.selectbox("LİG SEÇİN", list(LIGLER.keys()))
res = load_and_train(LIGLER[lig_adi])

if res[0] is not None:
    raw_df, m_ev, m_dep, m_ht_ev, m_ht_dep, le, takimlar = res
    c_s1, c_s2 = st.columns(2)
    ev = c_s1.selectbox("🏠 EV", takimlar)
    dep = c_s2.selectbox("🚀 DEP", takimlar)

    if st.button("📊 ANALİZİ BAŞLAT", use_container_width=True):
        c1, c2 = TAKIM_RENKLERI.get(ev, ("#1e293b", "#3b82f6"))
        ev_f = (sum([3 if (r['HomeTeam']==ev and r['FTHG']>r['FTAG']) or (r['AwayTeam']==ev and r['FTAG']>r['FTHG']) else 1 if r['FTHG']==r['FTAG'] else 0 for _, r in raw_df[(raw_df['HomeTeam']==ev) | (raw_df['AwayTeam']==ev)].tail(5).iterrows()])/15)
        dep_f = (sum([3 if (r['HomeTeam']==dep and r['FTHG']>r['FTAG']) or (r['AwayTeam']==dep and r['FTAG']>r['FTHG']) else 1 if r['FTHG']==r['FTAG'] else 0 for _, r in raw_df[(raw_df['HomeTeam']==dep) | (raw_df['AwayTeam']==dep)].tail(5).iterrows()])/15)
        
        g = [[le.transform([ev])[0], le.transform([dep])[0]]]
        fe, fd, he, hd = m_ev.predict(g)[0], m_dep.predict(g)[0], m_ht_ev.predict(g)[0], m_ht_dep.predict(g)[0]
        evw, ber, depw, u25, kg, htft, top3, surp = engine_v27(fe, fd, he, hd, ev_f, dep_f)

        # GÖRSEL SONUÇ (Hata Giderildi)
        st.markdown(f"""<div style="background:{c1}; color:white; padding:15px; border-radius:10px; text-align:center; border-bottom:5px solid {c2}">
        <h2 style="margin:0">{ev.upper()} vs {dep.upper()}</h2></div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("SKOR TAHMİNİ", f"{int(np.round(fe))}-{int(np.round(fd))}")
            st.write(f"**İY:** {int(np.round(he))}-{int(np.round(hd))}")
        with col2:
            st.write(f"**{ev}:** %{evw:.1f} | **BER:** %{ber:.1f} | **{dep}:** %{depw:.1f}")
            st.write(f"📈 2.5 ÜST: %{u25:.1f} | ⚽ KG VAR: %{kg:.1f}")

        st.subheader("🔮 HT/FT MATRİSİ (TAM LİSTE)")
        cols = st.columns(3)
        for i, (r, p) in enumerate(htft):
            cols[i%3].markdown(f"""<div style="background:#f1f5f9; padding:10px; border-radius:5px; text-align:center; margin-bottom:10px; border:1px solid {c1}"><b>{r}</b><br>%{p:.1f}</div>""", unsafe_allow_html=True)
        
        st.info(f"💣 SÜRPRİZ: {surp[0][0]}-{surp[0][1]} (%{surp[1]:.1f}) | 🚀 FORM: %{ev_f*100:.0f} vs %{dep_f*100:.0f}")
