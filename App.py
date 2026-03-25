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
import requests # Logoları çekmek için

warnings.filterwarnings('ignore')

# --- 1. PREMIUM UI CONFIG & CUSTOM CSS ---
st.set_page_config(page_title="Pro-Scout Master v33.0", layout="wide", initial_sidebar_state="collapsed")

# Takım Renkleri ve Logoları (Buraya daha fazla logo ekleyebilirsiniz)
TEAM_DATA = {
    'Galatasaray': {'colors': ('#890e10', '#fdb912'), 'logo': 'https://upload.wikimedia.org/wikipedia/commons/e/ea/Galatasaray_Sports_Club_Logo.svg'},
    'Fenerbahce': {'colors': ('#002e5d', '#fbda17'), 'logo': 'https://upload.wikimedia.org/wikipedia/tr/8/86/Fenerbah%C3%A7e_SK.png'},
    'Besiktas': {'colors': ('#000000', '#ffffff'), 'logo': 'https://upload.wikimedia.org/wikipedia/tr/1/1a/Be%C5%9Fikta%C5%9F_JK_logo.png'},
    'Trabzonspor': {'colors': ('#800000', '#00a1e1'), 'logo': 'https://upload.wikimedia.org/wikipedia/commons/e/e0/Trabzonspor_Logo.svg'},
    # Avrupa Devleri
    'Real Madrid': {'colors': ('#0047a0', '#ffffff'), 'logo': 'https://upload.wikimedia.org/wikipedia/en/5/56/Real_Madrid_CF_logo.svg'},
    'Barcelona': {'colors': ('#a50044', '#004d98'), 'logo': 'https://upload.wikimedia.org/wikipedia/en/4/47/FC_Barcelona_%28crest%29.svg'},
    'Man City': {'colors': ('#6caddf', '#1c2c5b'), 'logo': 'https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg'},
    'Arsenal': {'colors': ('#ef0107', '#061922'), 'logo': 'https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg'},
    'Liverpool': {'colors': ('#c8102e', '#00a1e1'), 'logo': 'https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg'},
    'Bayern Munich': {'colors': ('#dc052d', '#0066b2'), 'logo': 'https://upload.wikimedia.org/wikipedia/commons/1/1b/FC_Bayern_M%C3%BCnchen_logo_%282017%29.svg'}
}
DEFAULT_LOGO = 'https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg'

# UI Stilleri (Modern Card Layout)
st.markdown("""
<style>
    .stApp { background-color: #f0f2f5; }
    .scout-card {
        background-color: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;
    }
    .main-header {
        background: linear-gradient(90deg, #1e293b 0%, #334155 100%);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; margin-bottom: 25px;
    }
    .score-hero { font-size: 56px; font-weight: 800; color: #1e293b; margin: 0; }
    .iy-hero { font-size: 18px; font-weight: 700; color: #64748b; margin: 0; }
    .olasi-skor { background-color: #f1f5f9; padding: 10px; border-radius: 8px; margin: 5px 0; border: 1px solid #ddd; text-align:center;}
</style>
""", unsafe_allow_html=True)

# LİG VE VERİ MOTORU (Değişmedi)
LIGLER = {'Türkiye (Süper Lig)': 'T1', 'İngiltere (Premier Lig)': 'E0', 'İspanya (La Liga)': 'SP1', 'Almanya (Bundesliga)': 'D1', 'İtalya (Serie A)': 'I1', 'Fransa (Ligue 1)': 'F1', 'Hollanda (Eredivisie)': 'N1'}

@st.cache_data
def master_load_v33(lig_kodu):
    mega = pd.DataFrame()
    current_teams = []
    for s in ["2324", "2425", "2526"]:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{s}/{lig_kodu}.csv"
            mega = pd.concat([mega, pd.read_csv(url)], ignore_index=True)
            current_teams = sorted(pd.concat([mega['HomeTeam'], mega['AwayTeam']]).unique().tolist())
        except: continue
    if mega.empty: return None
    df = mega[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG']].dropna()
    le = LabelEncoder().fit(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    df['Ev_K'], df['Dep_K'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    X = df[['Ev_K', 'Dep_K']].values
    m_ev, m_dep = RandomForestRegressor(n_estimators=100).fit(X, df['FTHG']), RandomForestRegressor(n_estimators=100).fit(X, df['FTAG'])
    m_ht_ev, m_ht_dep = RandomForestRegressor(n_estimators=100).fit(X, df['HTHG']), RandomForestRegressor(n_estimators=100).fit(X, df['HTAG'])
    return mega, m_ev, m_dep, m_ht_ev, m_ht_dep, le, current_teams

# ANALİZ MOTORU (Top 8 Skor Eklendi)
def engine_v33(ev_b, dep_b, h_ev_b, h_dep_b, ev_f, dep_f):
    ev_b *= (1 + (ev_f - 0.5) * 0.3); dep_b *= (1 + (dep_f - 0.5) * 0.3)
    ev_p, dep_p = [poisson.pmf(i, ev_b) for i in range(6)], [poisson.pmf(j, dep_b) for j in range(6)]
    m_ft = np.outer(ev_p, dep_p)
    ev_w, ber, dep_w = np.sum(np.tril(m_ft, -1))*100, np.sum(np.diag(m_ft))*100, np.sum(np.triu(m_ft, 1))*100
    u15, u25, kg = sum(m_ft[i,j] for i in range(6) for j in range(6) if i+j > 1.5)*100, sum(m_ft[i,j] for i in range(6) for j in range(6) if i+j > 2.5)*100, (1 - poisson.pmf(0, ev_b)) * (1 - poisson.pmf(0, dep_b)) * 100
    m_ht = np.outer([poisson.pmf(i, h_ev_b) for i in range(4)], [poisson.pmf(j, h_dep_b) for j in range(4)])
    ht_r = {"1": np.sum(np.tril(m_ht, -1)), "0": np.sum(np.diag(m_ht)), "2": np.sum(np.triu(m_ht, 1))}
    ft_r = {"1": ev_w/100, "0": ber/100, "2": dep_w/100}
    htft = sorted([(f"{h}/{f}", ht_r[h]*ft_r[f]*100) for h in ['1','0','2'] for f in ['1','0','2']], key=lambda x: x[1], reverse=True)
    # Top 8 Skor Listesi
    full_scores = sorted([((i,j), m_ft[i,j]*100) for i in range(5) for j in range(5)], key=lambda x: x[1], reverse=True)
    top_scores = full_scores[:8]
    surp_scores = full_scores[10:13]
    return ev_w, ber, dep_w, u15, u25, kg, htft, top_scores, surp_scores

# --- 3. INFOGRAPHIC ---
def get_v33_infographic(ev, dep, ms, iy, htft, kg, u25, p1, p2):
    fig, ax = plt.subplots(figsize=(6, 9), facecolor=p1)
    ax.axis('off')
    plt.text(0.5, 0.92, "PRO-SCOUTMASTER v33.0", color='white', fontsize=22, ha='center', weight='bold')
    plt.text(0.5, 0.85, f"{ev.upper()} vs {dep.upper()}", color='white', fontsize=16, ha='center', bbox=dict(facecolor=p2, alpha=0.9))
    content = f"SCORE: {ms}\nHT: {iy}\nHT/FT: {htft}\nKG VAR: %{kg:.1f}\n2.5 ÜST: %{u25:.1f}"
    plt.text(0.5, 0.5, content, color='black', fontsize=18, ha='center', va='center', linespacing=2, bbox=dict(facecolor='white', boxstyle='round,pad=1', edgecolor=p2, lw=4))
    buf = BytesIO(); plt.savefig(buf, format="png", bbox_inches='tight', dpi=150); return buf.getvalue()

# --- 4. ULTIMATE MASTER UI ASSEMBLY ---
st.markdown("<div class='main-header'><h1>🏆 PRO-SCOUT MASTER v33.0</h1></div>", unsafe_allow_html=True)

# Sol Panel Konfigürasyon
st.sidebar.markdown("### ⚙️ AYARLAR")
lig_adi = st.sidebar.selectbox("🌍 Lig Seçin", list(LIGLER.keys()))
res_v33 = master_load_v33(LIGLER[lig_adi])

if res_v33:
    raw_df, m_ev, m_dep, m_ht_ev, m_ht_dep, le, takimlar = res_v33
    ev_t = st.sidebar.selectbox("🏠 Ev Sahibi", takimlar)
    dep_t = st.sidebar.selectbox("🚀 Deplasman", takimlar)
    btn_run = st.sidebar.button("📊 DERİN ANALİZİ BAŞLAT", use_container_width=True)

    if btn_run:
        # Renk ve Logo Yönetimi
        p1, p2 = TAKIM_RENKLERI.get(ev_t, {'colors': ("#1e293b", "#3b82f6")})['colors']
        ev_logo = TAKIM_RENKLERI.get(ev_t, {'logo': DEFAULT_LOGO})['logo']
        dep_logo = TAKIM_RENKLERI.get(dep_t, {'logo': DEFAULT_LOGO})['logo']

        # Form Hesapla
        ev_f = (sum([3 if (r['HomeTeam']==ev_t and r['FTHG']>r['FTAG']) or (r['AwayTeam']==ev_t and r['FTAG']>r['FTHG']) else 1 if r['FTHG']==r['FTAG'] else 0 for _, r in raw_df[(raw_df['HomeTeam']==ev_t) | (raw_df['AwayTeam']==ev_t)].tail(5).iterrows()])/15)
        dep_f = (sum([3 if (r['HomeTeam']==dep_t and r['FTHG']>r['FTAG']) or (r['AwayTeam']==dep_t and r['FTAG']>r['FTHG']) else 1 if raw_df.loc[_,'FTHG']==raw_df.loc[_,'FTAG'] else 0 for _, r in raw_df[(raw_df['HomeTeam']==dep_t) | (raw_df['AwayTeam']==dep_t)].tail(5).iterrows()])/15)
        
        g = [[le.transform([ev_t])[0], le.transform([dep_t])[0]]]
        fe, fd, he, hd = m_ev.predict(g)[0], m_dep.predict(g)[0], m_ht_ev.predict(g)[0], m_ht_dep.predict(g)[0]
        evw, ber, depw, u15, u25, kg, htft, top8, surp = engine_v33(fe, fd, he, hd, ev_f, dep_f)

        # UI: Takım Başlığı & Logolar (Dinamik Gradient)
        st.markdown(f"""<div style="background:linear-gradient(90deg, {p1} 0%, {p2} 100%); color:white; padding:20px; border-radius:15px; text-align:center; box-shadow: 0 4px 6px rgba(0,0,0,0.2); margin-bottom:20px;">
        <div style="display:flex; justify-content:center; align-items:center; gap:20px;">
        <img src="{ev_logo}" width="60" style="background:white; border-radius:50%; padding:5px;"/>
        <h2 style="margin:0; letter-spacing:1px;">{ev_t.upper()} vs {dep_t.upper()}</h2>
        <img src="{dep_logo}" width="60" style="background:white; border-radius:50%; padding:5px;"/>
        </div>
        <small>Form Momentum: %{ev_f*100:.0f} vs %{dep_f*100:.0f} | 2026 Season Data</small></div>""", unsafe_allow_html=True)

        # UI: Ana Sonuç Kartı
        col_main1, col_main2 = st.columns([1, 1.2])
        with col_main1:
            st.markdown(f"""<div class='scout-card' style='text-align:center; border-left: 8px solid {p1};'>
            <span style="font-size:14px; font-weight:700; color:#64748b;">🎯 TABELA TAHMİNİ</span>
            <div class='score-hero'>{int(np.round(fe))}-{int(np.round(fd))}</div>
            <div class='iy-hero'>İY: {int(np.round(he))}-{int(np.round(hd))}</div></div>""", unsafe_allow_html=True)
            
            # OLASI SKORLAR (TOP 8)
            st.markdown("<h4 style='color:#1e293b; border-bottom:2px solid #ddd; padding-bottom:5px;'>📊 EN OLASI SKORLAR (TOP 8)</h4>", unsafe_allow_html=True)
            olasi_cols1, olasi_cols2 = st.columns(2)
            for i, (skr, pr) in enumerate(top8):
                c = olasi_cols1 if i < 4 else olasi_cols2
                c.markdown(f"<div class='olasi-skor'><b>{skr[0]}-{skr[1]}</b><br><small>%{pr:.1f}</small></div>", unsafe_allow_html=True)

        with col_main2:
            st.markdown(f"""<div class='scout-card' style='border-left: 8px solid {p2};'>
            <h4 style='margin:0 0 10px 0; color:#1e293b;'>⚖️ MAÇ SONUCU OLASILIKLARI</h4>""", unsafe_allow_html=True)
            st.progress(evw/100, text=f"{ev_t}: %{evw:.1f}")
            st.progress(ber/100, text=f"🤝 Beraberlik: %{ber:.1f}")
            st.progress(depw/100, text=f"{dep_t}: %{depw:.1f}")
            st.markdown(f"<div style='background:#f8fafc; padding:10px; border-radius:8px; border:1px solid #ddd; margin-top:10px;'><b>GOL ANALİZİ:</b><br>2.5 ÜST: %{u25:.1f} | 1.5 ÜST: %{u15:.1f}<br><b>Karşılıklı Gol (KG VAR): %{kg:.1f}</b></div></div>", unsafe_allow_html=True)

        st.markdown("---")
        # UI: HT/FT MATRİSİ (Tam Liste Korundu)
        st.subheader("🔮 9'LU HT/FT TAM MATRİS ANALİZİ")
        ht_cols = st.columns(3)
        for i, (res, prob) in enumerate(htft):
            ht_cols[i%3].markdown(f"<div style='background:#f1f5f9; padding:12px; border-radius:10px; text-align:center; border:2px solid {p1}; margin-bottom:10px;'><b>{res}</b><br><small>%{prob:.1f}</small></div>", unsafe_allow_html=True)

        st.markdown("---")
        # UI: STRATEJİ DANIŞMANI (Değişmedi)
        st.subheader("💡 STRATEJİ & SÜRPRİZ DANIŞMANI")
        c_st1, c_st2 = st.columns(2)
        c_st1.info(f"🟢 **GÜVENLİ:** {'1.5 Üst (%{:.0f})'.format(u15) if u15 > 75 else 'Çifte Şans 1X' if evw+ber > 80 else 'Maçın gidişatını gör.'}")
        c_st1.warning(f"🟡 **ANA TERCİH:** {'2.5 Üst (%{:.0f})'.format(u25) if u25 > 60 else 'KG VAR (%{:.0f})'.format(kg) if kg > 55 else 'Taraf Bahsi (MS)'}")
        c_st2.error(f"💣 **SÜRPRİZ SKOR (DEĞERLİ):** {surp[0][0][0]}-{surp[0][0][1]} (%{surp[0][1]:.1f})")
        
        # UI: İNDİRME BUTONU
        report_data = get_v33_infographic(ev_t, dep_t, f"{int(np.round(fe))}-{int(np.round(fd))}", f"{int(np.round(he))}-{int(np.round(hd))}", htft[0][0], kg, u25, p1, p2)
        st.download_button("🖼️ KUPON GÖRSELİNİ İNDİR (PNG)", data=report_data, file_name=f"{ev_t}_{dep_t}_analiz.png", mime="image/png", use_container_width=True)
