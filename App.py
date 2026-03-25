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

# --- 1. PREMIUM UI & LOGO MAPPING ---
st.set_page_config(page_title="Pro-Scout Master v36.0", layout="wide")

# Logoları daha geniş bir isim havuzuyla eşleştiriyoruz (CSV'deki isimlere göre)
TEAM_DATA = {
    'Galatasaray': {'colors': ('#890e10', '#fdb912'), 'logo': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Galatasaray_Sports_Club_Logo.svg/300px-Galatasaray_Sports_Club_Logo.svg.png'},
    'Fenerbahce': {'colors': ('#002e5d', '#fbda17'), 'logo': 'https://upload.wikimedia.org/wikipedia/tr/8/86/Fenerbah%C3%A7e_SK.png'},
    'Besiktas': {'colors': ('#000000', '#ffffff'), 'logo': 'https://upload.wikimedia.org/wikipedia/tr/1/1a/Be%C5%9Fikta%C5%9F_JK_logo.png'},
    'Trabzonspor': {'colors': ('#800000', '#00a1e1'), 'logo': 'https://upload.wikimedia.org/wikipedia/tr/b/b1/Trabzonspor_logo_2018.png'},
    'Real Madrid': {'colors': ('#ffffff', '#0047a0'), 'logo': 'https://upload.wikimedia.org/wikipedia/en/thumb/5/56/Real_Madrid_CF_logo.svg/300px-Real_Madrid_CF_logo.svg.png'},
    'Barcelona': {'colors': ('#a50044', '#004d98'), 'logo': 'https://upload.wikimedia.org/wikipedia/en/thumb/4/47/FC_Barcelona_%28crest%29.svg/300px-FC_Barcelona_%28crest%29.svg.png'},
    'Man City': {'colors': ('#6caddf', '#1c2c5b'), 'logo': 'https://upload.wikimedia.org/wikipedia/en/thumb/e/eb/Manchester_City_FC_badge.svg/300px-Manchester_City_FC_badge.svg.png'},
    'Arsenal': {'colors': ('#ef0107', '#061922'), 'logo': 'https://upload.wikimedia.org/wikipedia/en/thumb/5/53/Arsenal_FC.svg/300px-Arsenal_FC.svg.png'},
    'Liverpool': {'colors': ('#c8102e', '#f6eb61'), 'logo': 'https://upload.wikimedia.org/wikipedia/en/thumb/0/0c/Liverpool_FC.svg/300px-Liverpool_FC.svg.png'},
    'Bayern Munich': {'colors': ('#dc052d', '#0066b2'), 'logo': 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/FC_Bayern_M%C3%BCnchen_logo_%282017%29.svg/300px-FC_Bayern_M%C3%BCnchen_logo_%282017%29.svg.png'}
}
DEFAULT_LOGO = 'https://img.icons8.com/color/96/football2.png'
LIGLER = {'Türkiye (Süper Lig)': 'T1', 'İngiltere (Premier Lig)': 'E0', 'İspanya (La Liga)': 'SP1', 'Almanya (Bundesliga)': 'D1', 'İtalya (Serie A)': 'I1', 'Fransa (Ligue 1)': 'F1', 'Hollanda (Eredivisie)': 'N1'}

# --- 2. 3 YILLIK ANALİZ MOTORU ---
@st.cache_data
def load_and_analyze_v36(lig_kodu):
    mega = pd.DataFrame()
    active_teams = []
    # 23/24, 24/25 ve 25/26 Sezonlarını kapsayan 3 yıllık analiz temeli
    for s in ["2324", "2425", "2526"]:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{s}/{lig_kodu}.csv"
            df_s = pd.read_csv(url)
            mega = pd.concat([mega, df_s], ignore_index=True)
            active_teams = sorted(pd.concat([df_s['HomeTeam'], df_s['AwayTeam']]).unique().tolist())
        except: continue
    
    if mega.empty: return None
    df = mega[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG']].dropna()
    le = LabelEncoder().fit(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    df['Ev_K'], df['Dep_K'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    X = df[['Ev_K', 'Dep_K']].values
    
    # Derin Öğrenme (Random Forest - 100 Trees)
    m_ev, m_dep = RandomForestRegressor(n_estimators=100).fit(X, df['FTHG']), RandomForestRegressor(n_estimators=100).fit(X, df['FTAG'])
    m_ht_ev, m_ht_dep = RandomForestRegressor(n_estimators=100).fit(X, df['HTHG']), RandomForestRegressor(n_estimators=100).fit(X, df['HTAG'])
    return mega, m_ev, m_dep, m_ht_ev, m_ht_dep, le, active_teams

def engine_v36(ev_b, dep_b, h_ev_b, h_dep_b, ev_f, dep_f):
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

# --- 3. INFOGRAPHIC ---
def draw_v36_card(ev, dep, ms, iy, htft, kg, u25, p1, p2):
    fig, ax = plt.subplots(figsize=(6, 9), facecolor=p1)
    ax.axis('off')
    plt.text(0.5, 0.94, "PRO-SCOUTMASTER v36.0", color='white', fontsize=22, ha='center', weight='bold')
    plt.text(0.5, 0.86, f"{ev.upper()} vs {dep.upper()}", color='white', fontsize=16, ha='center', bbox=dict(facecolor=p2, alpha=0.9))
    content = f"SCORE: {ms}\nHT: {iy}\nHT/FT: {htft}\nKG VAR: %{kg:.1f}\n2.5 ÜST: %{u25:.1f}"
    plt.text(0.5, 0.5, content, color='black', fontsize=18, ha='center', va='center', linespacing=2, bbox=dict(facecolor='white', boxstyle='round,pad=1', edgecolor=p2, lw=4))
    buf = BytesIO(); plt.savefig(buf, format="png", bbox_inches='tight', dpi=150); return buf.getvalue()

# --- 4. MASTER UI ---
st.markdown("<h2 style='text-align: center; color: #1e293b; margin-top: -30px;'>🏆 PRO-SCOUT MASTER v36.0</h2>", unsafe_allow_html=True)

# Seçim Alanı
lig_adi = st.selectbox("🌍 ANALİZ EDİLECEK LİG", list(LIGLER.keys()))
res_v36 = load_and_analyze_v36(LIGLER[lig_adi])

if res_v36:
    mega, m_ev, m_dep, m_ht_ev, m_ht_dep, le, active_teams = res_v36
    c_s1, c_s2 = st.columns(2)
    ev_t = c_s1.selectbox("🏠 EV SAHİBİ", active_teams)
    dep_t = c_s2.selectbox("🚀 DEPLASMAN", active_teams)

    if st.button("📊 DERİN ANALİZİ BAŞLAT", use_container_width=True):
        p1, p2 = TEAM_DATA.get(ev_t, {'colors': ("#1e293b", "#3b82f6")})['colors']
        ev_logo = TEAM_DATA.get(ev_t, {'logo': DEFAULT_LOGO})['logo']
        dep_logo = TEAM_DATA.get(dep_t, {'logo': DEFAULT_LOGO})['logo']

        # Form Hesaplama (3 Yıllık Veri Üzerinden Son 5 Maç)
        ev_f = (sum([3 if (r['HomeTeam']==ev_t and r['FTHG']>r['FTAG']) or (r['AwayTeam']==ev_t and r['FTAG']>r['FTHG']) else 1 if r['FTHG']==r['FTAG'] else 0 for _, r in mega[(mega['HomeTeam']==ev_t) | (mega['AwayTeam']==ev_t)].tail(5).iterrows()])/15)
        dep_f = (sum([3 if (r['HomeTeam']==dep_t and r['FTHG']>r['FTAG']) or (r['AwayTeam']==dep_t and r['FTAG']>r['FTHG']) else 1 if mega.loc[_,'FTHG']==mega.loc[_,'FTAG'] else 0 for _, r in mega[(mega['HomeTeam']==dep_t) | (mega['AwayTeam']==dep_t)].tail(5).iterrows()])/15)
        
        g = [[le.transform([ev_t])[0], le.transform([dep_t])[0]]]
        fe, fd, he, hd = m_ev.predict(g)[0], m_dep.predict(g)[0], m_ht_ev.predict(g)[0], m_ht_dep.predict(g)[0]
        evw, ber, depw, u15, u25, kg, htft, top8, surp = engine_v36(fe, fd, he, hd, ev_f, dep_f)

        # 1. BAŞLIK VE LOGO PANELİ (GÜVENLİ YÖNTEM)
        st.markdown(f"""<div style='background: linear-gradient(90deg, {p1} 0%, {p2} 100%); padding: 25px; border-radius: 15px; text-align: center; color: white; margin-bottom: 25px;'>
        <h1 style='margin: 0; font-size: 24px; text-transform: uppercase;'>{ev_t} vs {dep_t}</h1>
        <p style='margin: 0; opacity: 0.9;'>FORM: %{ev_f*100:.0f} - %{dep_f*100:.0f} | 3 YILLIK VERİ ANALİZİ</p></div>""", unsafe_allow_html=True)
        
        l_col1, l_col2, l_col3 = st.columns([1, 1, 1])
        with l_col1: st.image(ev_logo, width=120)
        with l_col2: st.markdown("<div style='text-align:center; padding-top:20px;'><h1 style='font-size:60px;'>VS</h1></div>", unsafe_allow_html=True)
        with l_col3: st.image(dep_logo, width=120)

        st.divider()

        # 2. ANA ANALİZ PANELİ
        col_main1, col_main2 = st.columns([1, 1.2])
        with col_main1:
            st.markdown(f"""<div style="background:white; padding:20px; border-radius:15px; border-left:10px solid {p1}; text-align:center; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
            <span style="font-weight:800; color:#64748b;">🎯 MS SKOR TAHMİNİ</span>
            <div style="font-size:55px; font-weight:900; color:#1e293b; margin:10px 0;">{int(np.round(fe))}-{int(np.round(fd))}</div>
            <div style="font-size:20px; font-weight:700; color:#64748b;">İY: {int(np.round(he))}-{int(np.round(hd))}</div></div>""", unsafe_allow_html=True)
            
            st.markdown("<h4 style='margin-top:20px; color:#1e293b;'>📊 EN OLASI 8 SKOR</h4>", unsafe_allow_html=True)
            sk_c1, sk_c2 = st.columns(2)
            for i, (sk, pr) in enumerate(top8):
                target = sk_c1 if i < 4 else sk_c2
                target.markdown(f"<div style='background:#f1f5f9; padding:8px; border-radius:8px; margin:4px 0; text-align:center; border:1px solid #ddd;'><b>{sk[0]}-{sk[1]}</b><br><small>%{pr:.1f}</small></div>", unsafe_allow_html=True)

        with col_main2:
            st.markdown(f"""<div style="background:white; padding:20px; border-radius:15px; border-left:10px solid {p2}; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="margin:0 0 15px 0;">⚖️ OLASILIK ANALİZİ</h4>""", unsafe_allow_html=True)
            st.progress(evw/100, text=f"{ev_t}: %{evw:.1f}")
            st.progress(ber/100, text=f"🤝 Beraberlik: %{ber:.1f}")
            st.progress(depw/100, text=f"{dep_t}: %{depw:.1f}")
            st.markdown(f"""<div style="background:#f8fafc; padding:12px; border-radius:10px; border:1px solid #ddd; margin-top:15px;">
            <b>GOL VERİLERİ:</b><br>
            2.5 ÜST: %{u25:.1f} | 1.5 ÜST: %{u15:.1f}<br>
            <b style="color:{p1}">KG VAR: %{kg:.1f}</b></div></div>""", unsafe_allow_html=True)

        st.divider()
        
        # 3. HT/FT VE STRATEJİ
        st.subheader("🔮 9'LU HT/FT TAM MATRİS")
        ht_cols = st.columns(3)
        for i, (res, prob) in enumerate(htft):
            ht_cols[i%3].markdown(f"<div style='background:#f1f5f9; padding:12px; border-radius:10px; text-align:center; border:2px solid {p1}; margin-bottom:12px;'><b>{res}</b><br>%{prob:.1f}</div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("💡 STRATEJİ DANIŞMANI")
        st.info(f"🟢 **GÜVENLİ:** {'1.5 Üst (%{:.0f})'.format(u15) if u15 > 75 else 'Çifte Şans 1X' if evw+ber > 80 else 'Maçın gidişatını izle.'}")
        st.warning(f"🟡 **ANA TERCİH:** {'2.5 Üst (%{:.0f})'.format(u25) if u25 > 62 else 'KG VAR (%{:.0f})'.format(kg) if kg > 58 else 'Taraf Bahsi (MS)'}")
        st.error(f"💣 **SÜRPRİZ SKOR:** {surp[0][0][0]}-{surp[0][0][1]} (%{surp[0][1]:.1f})")
        
        report = draw_v36_card(ev_t, dep_t, f"{int(np.round(fe))}-{int(np.round(fd))}", f"{int(np.round(he))}-{int(np.round(hd))}", htft[0][0], kg, u25, p1, p2)
        st.download_button("🖼️ KUPON GÖRSELİNİ İNDİR (PNG)", data=report, file_name=f"{ev_t}_analiz.png", mime="image/png", use_container_width=True)
else:
    st.error("Lig verileri yüklenemedi. İnternet bağlantısını kontrol edin.")
