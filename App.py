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

# --- 1. PREMIUM UI & TEAM COLORS ---
st.set_page_config(page_title="Pro-Scout v38.0", layout="wide")

TEAM_COLORS = {
    'Galatasaray': ('#890e10', '#fdb912'), 'Fenerbahce': ('#002e5d', '#fbda17'), 
    'Besiktas': ('#000000', '#ffffff'), 'Trabzonspor': ('#800000', '#00a1e1'),
    'Real Madrid': ('#ffffff', '#0047a0'), 'Barcelona': ('#a50044', '#004d98'),
    'Man City': ('#6caddf', '#1c2c5b'), 'Arsenal': ('#ef0107', '#061922'),
    'Liverpool': ('#c8102e', '#f6eb61'), 'Bayern Munich': ('#dc052d', '#0066b2')
}

LIGLER = {'Türkiye (Süper Lig)': 'T1', 'İngiltere (Premier Lig)': 'E0', 'İspanya (La Liga)': 'SP1', 'Almanya (Bundesliga)': 'D1', 'İtalya (Serie A)': 'I1', 'Fransa (Ligue 1)': 'F1', 'Hollanda (Eredivisie)': 'N1'}

# --- 2. 3 YILLIK VERİ VE EĞİTİM MOTORU ---
@st.cache_data
def master_load_v38(lig_kodu):
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

# ⚔️ GELİŞMİŞ REKABET DANIŞMANI (H2H Pro)
def rekabet_pro_analiz(df, ev, dep):
    h2h = df[((df['HomeTeam'] == ev) & (df['AwayTeam'] == dep)) | ((df['HomeTeam'] == dep) & (df['AwayTeam'] == ev))]
    if h2h.empty: return {"status": False, "msg": "Son 3 yılda resmi maç kaydı bulunamadı."}
    
    # Galibiyetler
    ev_gal = len(h2h[(h2h['HomeTeam'] == ev) & (h2h['FTHG'] > h2h['FTAG'])]) + len(h2h[(h2h['AwayTeam'] == ev) & (h2h['FTAG'] > h2h['FTHG'])])
    dep_gal = len(h2h[(h2h['HomeTeam'] == dep) & (h2h['FTHG'] > h2h['FTAG'])]) + len(h2h[(h2h['AwayTeam'] == dep) & (h2h['FTAG'] > h2h['FTHG'])])
    ber = len(h2h[h2h['FTHG'] == h2h['FTAG']])
    
    # Gol Analizi
    ev_gol = h2h[h2h['HomeTeam'] == ev]['FTHG'].sum() + h2h[h2h['AwayTeam'] == ev]['FTAG'].sum()
    dep_gol = h2h[h2h['HomeTeam'] == dep]['FTHG'].sum() + h2h[h2h['AwayTeam'] == dep]['FTAG'].sum()
    avg_gol = (ev_gol + dep_gol) / len(h2h)
    
    # Savunma Analizi (Clean Sheet)
    ev_cs = len(h2h[(h2h['HomeTeam'] == ev) & (h2h['FTAG'] == 0)]) + len(h2h[(h2h['AwayTeam'] == ev) & (h2h['FTHG'] == 0)])
    
    res = {
        "status": True,
        "ev_w": ev_gal, "dep_w": dep_gal, "draw": ber,
        "ev_g": int(ev_gol), "dep_g": int(dep_gol),
        "avg": round(avg_gol, 2), "ev_cs": ev_cs,
        "total": len(h2h)
    }
    return res

def engine_v38(ev_b, dep_b, h_ev_b, h_dep_b, ev_f, dep_f):
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
def draw_v38_card(ev, dep, ms, iy, htft, kg, u25, p1, p2):
    fig, ax = plt.subplots(figsize=(6, 9), facecolor=p1)
    ax.axis('off')
    plt.text(0.5, 0.94, "PRO-SCOUTMASTER v38.0", color='white', fontsize=22, ha='center', weight='bold')
    plt.text(0.5, 0.86, f"{ev.upper()} vs {dep.upper()}", color='white', fontsize=16, ha='center', bbox=dict(facecolor=p2, alpha=0.9))
    content = f"SKOR: {ms}\nİY: {iy}\nHT/FT: {htft}\nKG VAR: %{kg:.1f}\n2.5 ÜST: %{u25:.1f}"
    plt.text(0.5, 0.5, content, color='black', fontsize=18, ha='center', va='center', linespacing=2, bbox=dict(facecolor='white', boxstyle='round,pad=1', edgecolor=p2, lw=4))
    buf = BytesIO(); plt.savefig(buf, format="png", bbox_inches='tight', dpi=150); return buf.getvalue()

# --- 4. MASTER UI ---
st.markdown("<h2 style='text-align:center;'>🏆 PRO-SCOUT MASTER v38.0</h2>", unsafe_allow_html=True)

lig_box = st.selectbox("🌍 ANALİZ EDİLECEK LİG", list(LIGLER.keys()))
res = master_load_v38(LIGLER[lig_box])

if res:
    mega, m_ev, m_dep, m_ht_ev, m_ht_dep, le, takimlar = res
    c1, c2 = st.columns(2)
    ev_t = c1.selectbox("🏠 EV SAHİBİ", takimlar)
    dep_t = c2.selectbox("🚀 DEPLASMAN", takimlar)

    if st.button("📊 DERİN ANALİZİ BAŞLAT", use_container_width=True):
        p1, p2 = TEAM_COLORS.get(ev_t, ("#1e293b", "#3b82f6"))
        
        # Analiz Verileri
        ev_f = (sum([3 if (r['HomeTeam']==ev_t and r['FTHG']>r['FTAG']) or (r['AwayTeam']==ev_t and r['FTAG']>r['FTHG']) else 1 if r['FTHG']==r['FTAG'] else 0 for _, r in mega[(mega['HomeTeam']==ev_t) | (mega['AwayTeam']==ev_t)].tail(5).iterrows()])/15)
        dep_f = (sum([3 if (r['HomeTeam']==dep_t and r['FTHG']>r['FTAG']) or (r['AwayTeam']==dep_t and r['FTAG']>r['FTHG']) else 1 if mega.loc[_,'FTHG']==mega.loc[_,'FTAG'] else 0 for _, r in mega[(mega['HomeTeam']==dep_t) | (mega['AwayTeam']==dep_t)].tail(5).iterrows()])/15)
        h2h = rekabet_pro_analiz(mega, ev_t, dep_t)

        # UI: PREMIUM HEADER
        st.markdown(f"""<div style="background:linear-gradient(135deg, {p1} 0%, {p2} 100%); color:white; padding:30px; border-radius:15px; text-align:center; box-shadow:0 4px 15px rgba(0,0,0,0.2); margin-bottom:25px;">
        <h1 style="margin:0; font-size:32px; letter-spacing:2px; text-transform:uppercase;">{ev_t} - {dep_t}</h1>
        <p style="margin-top:10px; font-weight:700;">FORM: %{ev_f*100:.0f} vs %{dep_f*100:.0f} | v38.0 PRO ELITE</p></div>""", unsafe_allow_html=True)

        # UI: GELİŞMİŞ REKABET KARTI (H2H Pro)
        st.markdown("### ⚔️ REKABET DANIŞMANI (H2H PRO)")
        if h2h['status']:
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            r_col1.metric("Maç Sayısı", h2h['total'])
            r_col2.metric(f"{ev_t} Gal.", h2h['ev_w'])
            r_col3.metric(f"{dep_t} Gal.", h2h['dep_w'])
            r_col4.metric("Beraberlik", h2h['draw'])
            
            st.markdown(f"""<div style='background:white; padding:15px; border-radius:10px; border-left:5px solid {p1}; margin-bottom:20px;'>
            <b>📊 Detaylı Rekabet Notları:</b><br>
            • Bu iki takım arasındaki maçlarda ortalama <b>{h2h['avg']} gol</b> atılıyor.<br>
            • {ev_t} bu rekabette toplam <b>{h2h['ev_g']} gol</b> atarken, {dep_t} <b>{h2h['dep_g']} golle</b> karşılık verdi.<br>
            • {ev_t}, rakibine karşı oynadığı maçların <b>{h2h['ev_cs']} tanesinde</b> kalesini gole kapattı.</div>""", unsafe_allow_html=True)
        else:
            st.info(h2h['msg'])

        col_main1, col_main2 = st.columns([1, 1.2])
        with col_main1:
            st.markdown(f"""<div style="background:white; padding:20px; border-radius:15px; border-left:10px solid {p1}; text-align:center; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
            <span style="font-weight:800; color:#64748b;">🎯 MS SKOR TAHMİNİ</span>
            <div style="font-size:55px; font-weight:900; color:#1e293b; margin:10px 0;">{int(np.round(m_ev.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0]))}-{int(np.round(m_dep.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0]))}</div>
            <div style="font-size:20px; font-weight:700; color:#64748b;">İY: {int(np.round(m_ht_ev.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0]))}-{int(np.round(m_ht_dep.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0]))}</div></div>""", unsafe_allow_html=True)
            
            evw, ber, depw, u15, u25, kg, htft, top8, surp = engine_v38(m_ev.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0], m_dep.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0], m_ht_ev.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0], m_ht_dep.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0], ev_f, dep_f)
            st.markdown("<h4 style='margin-top:20px; border-bottom:2px solid #ddd;'>📊 EN OLASI 8 SKOR</h4>", unsafe_allow_html=True)
            sk_c1, sk_c2 = st.columns(2)
            for i, (sk, pr) in enumerate(top8):
                target = sk_c1 if i < 4 else sk_c2
                target.markdown(f"<div style='background:#f1f5f9; padding:10px; border-radius:8px; margin:4px 0; text-align:center; border:1px solid #cbd5e1;'><b>{sk[0]}-{sk[1]}</b><br><small>%{pr:.1f}</small></div>", unsafe_allow_html=True)

        with col_main2:
            st.markdown(f"""<div style="background:white; padding:20px; border-radius:15px; border-left:10px solid {p2}; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="margin:0 0 15px 0; color:#1e293b;">⚖️ OLASILIK ANALİZİ</h4>""", unsafe_allow_html=True)
            st.progress(evw/100, text=f"{ev_t}: %{evw:.1f}")
            st.progress(ber/100, text=f"🤝 Beraberlik: %{ber:.1f}")
            st.progress(depw/100, text=f"{dep_t}: %{depw:.1f}")
            st.markdown(f"""<div style="background:#f8fafc; padding:15px; border-radius:10px; border:1px solid #ddd; margin-top:15px;">
            <b>GOL VERİLERİ:</b><br>
            2.5 ÜST: %{u25:.1f} | 1.5 ÜST: %{u15:.1f}<br>
            <b style="color:{p1}">KG VAR: %{kg:.1f}</b></div></div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("🔮 9'LU HT/FT TAM MATRİS")
        ht_cols = st.columns(3)
        for i, (res, prob) in enumerate(htft):
            ht_cols[i%3].markdown(f"<div style='background:#f1f5f9; padding:12px; border-radius:10px; text-align:center; border:2.2px solid {p1}; margin-bottom:12px;'><b>{res}</b><br><small>%{prob:.1f}</small></div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("💡 STRATEJİ DANIŞMANI")
        st.success(f"🟢 **GÜVENLİ:** {'1.5 Üst (%{:.0f})'.format(u15) if u15 > 75 else 'Çifte Şans 1X' if evw+ber > 80 else 'Maçın gidişatını bekle.'}")
        st.warning(f"🟡 **ANA TERCİH:** {'2.5 Üst (%{:.0f})'.format(u25) if u25 > 62 else 'KG VAR (%{:.0f})'.format(kg) if kg > 58 else 'Taraf Bahsi (MS)'}")
        st.error(f"💣 **SÜRPRİZ SKOR:** {surp[0][0][0]}-{surp[0][0][1]} (%{surp[0][1]:.1f})")
        
        report = draw_v38_card(ev_t, dep_t, "...", "...", htft[0][0], kg, u25, p1, p2)
        st.download_button("🖼️ KUPON GÖRSELİNİ İNDİR (PNG)", data=report, file_name=f"{ev_t}_analiz.png", mime="image/png", use_container_width=True)
