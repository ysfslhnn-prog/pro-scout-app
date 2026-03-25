import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.stats import poisson
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --- 1. MODERN TASARIM VE TEMA ---
st.set_page_config(page_title="Pro-Scout Mobile", layout="wide")

TAKIM_RENKLERI = {
    'Galatasaray': ('#fdb912', '#890e10'), 'Fenerbahce': ('#002e5d', '#fbda17'), 
    'Besiktas': ('#000000', '#ffffff'), 'Trabzonspor': ('#800000', '#00a1e1'),
    'Real Madrid': ('#ffffff', '#3e3181'), 'Barcelona': ('#a50044', '#004d98'),
    'Man City': ('#6caddf', '#1c2c5b'), 'Arsenal': ('#ef0107', '#061922'),
    'Liverpool': ('#c8102e', '#00b2a9'), 'Bayern Munich': ('#dc052d', '#0066b2')
}

LIGLER = {'Türkiye (Süper Lig)': 'T1', 'İngiltere (Premier Lig)': 'E0', 'İspanya (La Liga)': 'SP1', 'Almanya (Bundesliga)': 'D1', 'İtalya (Serie A)': 'I1', 'Fransa (Ligue 1)': 'F1', 'Hollanda (Eredivisie)': 'N1'}

# --- 2. VERİ VE EĞİTİM MOTORU ---
@st.cache_data
def master_load_and_train(lig_kodu):
    mega = pd.DataFrame()
    for s in ["2324", "2425", "2526"]:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{s}/{lig_kodu}.csv"
            mega = pd.concat([mega, pd.read_csv(url)], ignore_index=True)
        except: continue
    
    if mega.empty: return None
    
    df = mega[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG']].dropna()
    le = LabelEncoder().fit(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    df['Ev_K'], df['Dep_K'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    X = df[['Ev_K', 'Dep_K']].values
    
    # Random Forest Modelleri
    m_ev = RandomForestRegressor(n_estimators=100).fit(X, df['FTHG'])
    m_dep = RandomForestRegressor(n_estimators=100).fit(X, df['FTAG'])
    m_ht_ev = RandomForestRegressor(n_estimators=100).fit(X, df['HTHG'])
    m_ht_dep = RandomForestRegressor(n_estimators=100).fit(X, df['HTAG'])
    
    return mega, m_ev, m_dep, m_ht_ev, m_ht_dep, le, sorted(le.classes_)

# --- 3. ANALİZ MOTORU ---
def engine_v28(ev_b, dep_b, h_ev_b, h_dep_b, ev_f, dep_f):
    ev_b *= (1 + (ev_f - 0.5) * 0.3)
    dep_b *= (1 + (dep_f - 0.5) * 0.3)
    
    # MS Olasılıklar
    ev_p, dep_p = [poisson.pmf(i, ev_b) for i in range(6)], [poisson.pmf(j, dep_b) for j in range(6)]
    m_ft = np.outer(ev_p, dep_p)
    ev_w, ber, dep_w = np.sum(np.tril(m_ft, -1))*100, np.sum(np.diag(m_ft))*100, np.sum(np.triu(m_ft, 1))*100
    
    # Alt/Üst ve KG
    u15, u25 = sum(m_ft[i,j] for i in range(6) for j in range(6) if i+j > 1.5)*100, sum(m_ft[i,j] for i in range(6) for j in range(6) if i+j > 2.5)*100
    kg = (1 - poisson.pmf(0, ev_b)) * (1 - poisson.pmf(0, dep_b)) * 100
    
    # 9'lu HT/FT MATRİSİ
    m_ht = np.outer([poisson.pmf(i, h_ev_b) for i in range(4)], [poisson.pmf(j, h_dep_b) for j in range(4)])
    ht_r = {"1": np.sum(np.tril(m_ht, -1)), "0": np.sum(np.diag(m_ht)), "2": np.sum(np.triu(m_ht, 1))}
    ft_r = {"1": ev_w/100, "0": ber/100, "2": dep_w/100}
    htft = sorted([(f"{h}/{f}", ht_r[h]*ft_r[f]*100) for h in ['1','0','2'] for f in ['1','0','2']], key=lambda x: x[1], reverse=True)
    
    skr = sorted([((i,j), m_ft[i,j]*100) for i in range(5) for j in range(5)], key=lambda x: x[1], reverse=True)
    return ev_w, ber, dep_w, u15, u25, kg, htft, skr[:3], skr[4:7]

# --- 4. UI ASSEMBLY ---
st.markdown("<h1 style='text-align: center; color: #1e293b;'>🏆 PRO-SCOUT MASTER</h1>", unsafe_allow_html=True)

lig_adi = st.selectbox("🌍 LİG SEÇİN", list(LIGLER.keys()))
res = master_load_and_train(LIGLER[lig_adi])

if res:
    raw_df, m_ev, m_dep, m_ht_ev, m_ht_dep, le, takimlar = res
    c1, c2 = st.columns(2)
    ev_t = c1.selectbox("🏠 EV SAHİBİ", takimlar)
    dep_t = c2.selectbox("🚀 DEPLASMAN", takimlar)

    if st.button("📊 ANALİZİ BAŞLAT", use_container_width=True):
        primary, secondary = TAKIM_RENKLERI.get(ev_t, ("#1e293b", "#3b82f6"))
        
        # Form Hesaplama
        def get_form(t):
            maclar = raw_df[(raw_df['HomeTeam']==t) | (raw_df['AwayTeam']==t)].tail(5)
            p = sum([3 if (r['HomeTeam']==t and r['FTHG']>r['FTAG']) or (r['AwayTeam']==t and r['FTAG']>r['FTHG']) else 1 if r['FTHG']==r['FTAG'] else 0 for _, r in maclar.iterrows()])
            return p / 15
        
        ev_f, dep_f = get_form(ev_t), get_form(dep_t)
        g = [[le.transform([ev_t])[0], le.transform([dep_t])[0]]]
        fe, fd, he, hd = m_ev.predict(g)[0], m_dep.predict(g)[0], m_ht_ev.predict(g)[0], m_ht_dep.predict(g)[0]
        evw, ber, depw, u15, u25, kg, htft, top3, surp = engine_v28(fe, fd, he, hd, ev_f, dep_f)

        # UI KARTLARI
        st.markdown(f"""<div style="background:{primary}; color:white; padding:20px; border-radius:15px; text-align:center; border-bottom:8px solid {secondary}; margin-bottom:20px;">
        <h2 style="margin:0; letter-spacing:1px;">{ev_t.upper()} vs {dep_t.upper()}</h2>
        <small>Form: %{ev_f*100:.0f} vs %{dep_f*100:.0f}</small></div>""", unsafe_allow_html=True)

        col_skor, col_prob = st.columns([1, 1.2])
        with col_skor:
            st.markdown(f"""<div style="background:white; padding:15px; border-radius:12px; border:1px solid #ddd; text-align:center;">
            <span style="font-size:12px; font-weight:bold; color:#64748b;">SKOR TAHMİNİ</span>
            <div style="font-size:48px; font-weight:800; color:{primary};">{int(np.round(fe))}-{int(np.round(fd))}</div>
            <div style="font-weight:700; color:#64748b;">İY: {int(np.round(he))}-{int(np.round(hd))}</div></div>""", unsafe_allow_html=True)
            
            st.markdown("<br><b>🎯 EN OLASI SKORLAR:</b>", unsafe_allow_html=True)
            for s in top3: st.write(f"• {s[0][0]}-{s[0][1]} (%{s[1]:.1f})")

        with col_prob:
            st.write("**MAÇ SONUCU İHTİMALLERİ**")
            st.progress(evw/100, text=f"{ev_t}: %{evw:.1f}")
            st.progress(ber/100, text=f"Beraberlik: %{ber:.1f}")
            st.progress(depw/100, text=f"{dep_t}: %{depw:.1f}")
            
            st.markdown(f"""<div style="margin-top:15px; display:flex; gap:10px;">
            <span style="background:#dbeafe; color:#1e40af; padding:5px 10px; border-radius:5px; font-weight:700;">2.5 ÜST: %{u25:.1f}</span>
            <span style="background:#d1fae5; color:#065f46; padding:5px 10px; border-radius:5px; font-weight:700;">KG VAR: %{kg:.1f}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        
        # HT/FT MATRİSİ (TAM 9 SENARYO - KIRPILMADI)
        st.subheader("🔮 HT/FT TAM MATRİS ANALİZİ")
        ht_cols = st.columns(3)
        for i, (res, prob) in enumerate(htft):
            ht_cols[i%3].markdown(f"""<div style="background:#f8fafc; padding:12px; border-radius:8px; text-align:center; border:1px solid {primary}; margin-bottom:10px;">
            <b style="color:{primary};">{res}</b><br><small>%{prob:.1f}</small></div>""", unsafe_allow_html=True)

        st.markdown("---")
        
        # STRATEJİ VE SÜRPRİZ
        st.subheader("💡 STRATEJİ DANIŞMANI")
        st.info(f"🟢 **Düşük Risk:** {'1.5 Üst (%{:.0f})'.format(u15) if u15 > 75 else 'Çifte Şans 1X' if evw+ber > 80 else 'Maçın başını izle.'}")
        st.warning(f"🟡 **Değerli Tahmin:** {'2.5 Üst (%{:.0f})'.format(u25) if u25 > 60 else 'Karşılıklı Gol (%{:.0f})'.format(kg) if kg > 55 else 'Taraf Bahsi (MS)'}")
        st.error(f"💣 **SÜRPRİZ SKOR:** {surp[0][0][0]}-{surp[0][0][1]} (%{surp[0][1]:.1f})")

else:
    st.error("Lig verileri yüklenemedi. Lütfen interneti kontrol edin.")
