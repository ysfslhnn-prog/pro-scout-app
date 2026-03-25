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

# --- 1. UI & GLOBAL LEAGUES ARCHITECTURE ---
st.set_page_config(page_title="Pro-Scout v51.0 Ultimate", layout="wide")

TEAM_COLORS = {
    'Galatasaray': ('#890e10', '#fdb912'), 'Fenerbahce': ('#002e5d', '#fbda17'), 
    'Besiktas': ('#000000', '#ffffff'), 'Trabzonspor': ('#800000', '#00a1e1')
}

# TÜM LİGLER VE ALT LİGLER EKSİKSİZ
LIGLER = {
    '🇹🇷 Türkiye (Süper Lig)': 'T1', '🇹🇷 Türkiye (1. Lig)': 'T2',
    '🏴󠁧󠁢󠁥󠁮󠁧󠁿 İngiltere (Premier)': 'E0', '🏴󠁧󠁢󠁥󠁮󠁧󠁿 İngiltere (Championship)': 'E1',
    '🏴󠁧󠁢󠁥󠁮󠁧󠁿 İngiltere (League 1)': 'E2', '🏴󠁧󠁢󠁥󠁮󠁧󠁿 İngiltere (League 2)': 'E3',
    '🇪🇸 İspanya (La Liga)': 'SP1', '🇪🇸 İspanya (Segunda)': 'SP2',
    '🇩🇪 Almanya (Bundesliga 1)': 'D1', '🇩🇪 Almanya (Bundesliga 2)': 'D2',
    '🇮🇹 İtalya (Serie A)': 'I1', '🇮🇹 İtalya (Serie B)': 'I2',
    '🇫🇷 Fransa (Ligue 1)': 'F1', '🇫🇷 Fransa (Ligue 2)': 'F2',
    '🇳🇱 Hollanda (Eredivisie)': 'N1', '🇧🇪 Belçika (Jupiler Pro)': 'B1',
    '🇵🇹 Portekiz (Primeira)': 'P1', '🏴󠁧󠁢󠁳󠁣󠁴󠁿 İskoçya (Prem)': 'SC0',
    '🇬🇷 Yunanistan (Süper Lig)': 'G1', '🇦🇹 Avusturya (Bundesliga)': 'AUT',
    '🇩🇰 Danimarka (Superliga)': 'DNK', '🇨🇭 İsviçre (Super League)': 'SWZ'
}

# --- 2. CORE ENGINE & DATA (3 YEARS + PUAN DURUMU) ---
@st.cache_data
def master_load_v51(lig_kodu):
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
    
    # Canlı Puan Durumu Oluşturucu
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
    df = mega[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG']].dropna()
    le = LabelEncoder().fit(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    X = df[['Ev_K', 'Dep_K']] = np.array([[le.transform([r['HomeTeam']])[0], le.transform([r['AwayTeam']])[0]] for _, r in df.iterrows()])
    m_ev = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['FTHG'])
    m_dep = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['FTAG'])
    m_ht_ev = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['HTHG'])
    m_ht_dep = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['HTAG'])
    
    return mega, m_ev, m_dep, m_ht_ev, m_ht_dep, le, active_2026, standings

def rekabet_pro_v51(df, ev, dep):
    h2h = df[((df['HomeTeam'] == ev) & (df['AwayTeam'] == dep)) | ((df['HomeTeam'] == dep) & (df['AwayTeam'] == ev))]
    if h2h.empty: return {"status": False, "msg": "Rekabet kaydı bulunamadı."}
    ev_w = len(h2h[(h2h['HomeTeam'] == ev) & (h2h['FTHG'] > h2h['FTAG'])]) + len(h2h[(h2h['AwayTeam'] == ev) & (h2h['FTAG'] > h2h['FTHG'])])
    dep_w = len(h2h[(h2h['HomeTeam'] == dep) & (h2h['FTHG'] > h2h['FTAG'])]) + len(h2h[(h2h['AwayTeam'] == dep) & (h2h['FTAG'] > h2h['FTHG'])])
    draw = len(h2h[h2h['FTHG'] == h2h['FTAG']])
    avg = round((h2h['FTHG'].sum() + h2h['FTAG'].sum())/len(h2h), 2)
    ev_cs = len(h2h[(h2h['HomeTeam'] == ev) & (h2h['FTAG'] == 0)]) + len(h2h[(h2h['AwayTeam'] == ev) & (h2h['FTHG'] == 0)])
    return {"status": True, "total": len(h2h), "ev_w": ev_w, "dep_w": dep_w, "draw": draw, "avg": avg, "ev_cs": ev_cs}

def engine_v51(ev_b, dep_b, h_ev_b, h_dep_b, ev_f, dep_f):
    ev_b *= (1 + (ev_f - 0.5) * 0.4); dep_b *= (1 + (dep_f - 0.5) * 0.4)
    ev_p, dep_p = [poisson.pmf(i, ev_b) for i in range(7)], [poisson.pmf(j, dep_b) for j in range(7)]
    m_ft = np.outer(ev_p, dep_p)
    ev_w, ber, dep_w = np.sum(np.tril(m_ft, -1))*100, np.sum(np.diag(m_ft))*100, np.sum(np.triu(m_ft, 1))*100
    u15, u25, kg = sum(m_ft[i,j] for i in range(7) for j in range(7) if i+j > 1.5)*100, sum(m_ft[i,j] for i in range(7) for j in range(7) if i+j > 2.5)*100, (1 - ev_p[0]) * (1 - dep_p[0]) * 100
    m_ht = np.outer([poisson.pmf(i, h_ev_b) for i in range(4)], [poisson.pmf(j, h_dep_b) for j in range(4)])
    ht_r = {"1": np.sum(np.tril(m_ht, -1)), "0": np.sum(np.diag(m_ht)), "2": np.sum(np.triu(m_ht, 1))}
    htft = sorted([(f"{h}/{f}", ht_r[h]*( (ev_w/100) if f=='1' else (ber/100) if f=='0' else (dep_w/100))*100) for h in ['1','0','2'] for f in ['1','0','2']], key=lambda x: x[1], reverse=True)
    full_skr = sorted([((i,j), m_ft[i,j]*100) for i in range(5) for j in range(5)], key=lambda x: x[1], reverse=True)
    return ev_w, ber, dep_w, u15, u25, kg, htft, full_skr[:8], full_skr[10:13]

# --- 3. UI DASHBOARD ---
st.markdown("<h1 style='text-align:center;'>💎 PRO-SCOUT ULTIMATE v51.0</h1>", unsafe_allow_html=True)
tab_analiz, tab_puan, tab_banko = st.tabs(["📊 DERİN ANALİZ", "📈 CANLI PUAN DURUMU", "🔍 GÜNÜN BANKOLARI"])

res = master_load_v51(LIGLER[st.sidebar.selectbox("🌍 LİG SEÇİN", list(LIGLER.keys()))])

if res:
    mega, m_ev, m_dep, m_ht_ev, m_ht_dep, le, active_teams, table = res
    
    with tab_puan:
        st.markdown("### 🏟️ 2026 GÜNCEL PUAN DURUMU")
        st.table(table)

    with tab_analiz:
        c1, c2 = st.columns(2)
        ev_t, dep_t = c1.selectbox("🏠 EV", active_teams), c2.selectbox("🚀 DEP", active_teams)

        if st.button("📊 EKSİKSİZ ANALİZİ BAŞLAT", use_container_width=True):
            p1, p2 = TEAM_COLORS.get(ev_t, ("#1e293b", "#3b82f6"))
            
            # Form Karnesi
            def get_team_info(team):
                t_matches = mega[(mega['HomeTeam'] == team) | (mega['AwayTeam'] == team)].tail(5)
                points = sum([3 if (r['HomeTeam']==team and r['FTHG']>r['FTAG']) or (r['AwayTeam']==team and r['FTAG']>r['FTHG']) else 1 if r['FTHG']==r['FTAG'] else 0 for _, r in t_matches.iterrows()])
                scored = t_matches[t_matches['HomeTeam']==team]['FTHG'].sum() + t_matches[t_matches['AwayTeam']==team]['FTAG'].sum()
                conceded = t_matches[t_matches['HomeTeam']==team]['FTAG'].sum() + t_matches[t_matches['AwayTeam']==team]['FTHG'].sum()
                return points/15, scored, conceded

            ev_f, ev_gs, ev_gc = get_team_info(ev_t)
            dep_f, dep_gs, dep_gc = get_team_info(dep_t)
            
            fe, fd = m_ev.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0], m_dep.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0]
            he, hd = m_ht_ev.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0], m_ht_dep.predict([[le.transform([ev_t])[0], le.transform([dep_t])[0]]])[0]
            evw, ber, depw, u15, u25, kg, htft, top8, surp = engine_v51(fe, fd, he, hd, ev_f, dep_f)
            h2h = rekabet_pro_v51(mega, ev_t, dep_t)

            if any(p > 80 for p in [evw, ber, depw, u15, u25, (evw+ber)]):
                st.balloons()
                st.markdown(f"<div style='background:#fef3c7; color:#92400e; padding:15px; border-radius:10px; text-align:center; font-weight:bold; border:2px solid #f59e0b;'>⭐ PREMIUM TERCİH: 0.8 GÜVEN BARAJI GEÇİLDİ!</div>", unsafe_allow_html=True)

            st.markdown(f"<div style='background:linear-gradient(90deg, {p1} 0%, {p2} 100%); color:white; padding:30px; border-radius:15px; text-align:center;'><h2>{ev_t} - {dep_t}</h2></div>", unsafe_allow_html=True)
            
            k1, k2 = st.columns(2)
            k1.info(f"🏠 **{ev_t} Karnesi (Son 5 Maç):** {int(ev_f*15)} Puan | {ev_gs} G.Atıldı / {ev_gc} G.Yenildi")
            k2.success(f"🚀 **{dep_t} Karnesi (Son 5 Maç):** {int(dep_f*15)} Puan | {dep_gs} G.Atıldı / {dep_gc} G.Yenildi")

            # Grafik
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.pie([evw, ber, depw], labels=[ev_t, 'Beraberlik', dep_t], autopct='%1.1f%%', colors=[p1, '#cbd5e1', p2], startangle=90)
            ax.axis('equal'); st.pyplot(fig)

            cm1, cm2 = st.columns([1, 1.2])
            with cm1:
                st.markdown(f"<div style='background:white; padding:20px; border-radius:15px; border-left:10px solid {p1}; text-align:center;'>🎯 <b>SKOR: {int(np.round(fe))}-{int(np.round(fd))}</b><br><small>İY: {int(np.round(he))}-{int(np.round(hd))}</small></div>", unsafe_allow_html=True)
                st.write("### 📊 EN OLASI 8 SKOR")
                sk_c1, sk_c2 = st.columns(2)
                for i, (sk, pr) in enumerate(top8):
                    (sk_c1 if i < 4 else sk_c2).markdown(f"<div style='background:#f1f5f9; padding:8px; border-radius:8px; margin:2px; text-align:center;'><b>{sk[0]}-{sk[1]}</b> (%{pr:.1f})</div>", unsafe_allow_html=True)

            with cm2:
                st.write("### ⚖️ OLASILIK ANALİZİ")
                st.progress(evw/100, text=f"{ev_t}: %{evw:.1f} {'⭐' if evw > 80 else ''}")
                st.progress(ber/100, text=f"🤝 Beraberlik: %{ber:.1f}")
                st.progress(depw/100, text=f"{dep_t}: %{depw:.1f}")
                st.markdown(f"<div style='background:#f8fafc; padding:15px; border-radius:10px; border:1px solid #ddd; margin-top:10px; text-align:center;'>📈 1.5 ÜST: %{u15:.1f} | 🔥 2.5 ÜST: %{u25:.1f}<br><b>⚽ KARŞILIKLI GOL: %{kg:.1f}</b></div>", unsafe_allow_html=True)

            st.divider()
            st.subheader("⚔️ H2H REKABET İSTATİSTİKLERİ")
            if h2h['status']:
                r_c1, r_c2, r_c3 = st.columns(3)
                r_c1.metric("Toplam Maç", h2h['total'])
                r_c2.metric(f"{ev_t} Gal.", h2h['ev_w'])
                r_c3.metric(f"{dep_t} Gal.", h2h['dep_w'])
                st.info(f"Bu rekabette maç başı ortalama **{h2h['avg']} gol** atılıyor. {ev_t}, **{h2h['ev_cs']} maçta** gol yemedi.")
            else:
                st.warning("Son 3 yılda rekabet kaydı yok.")

            st.divider()
            st.subheader("🔮 9'LU HT/FT MATRİSİ")
            ht_cols = st.columns(3)
            for i, (res, prob) in enumerate(htft):
                ht_cols[i%3].markdown(f"<div style='background:#f1f5f9; padding:10px; border-radius:8px; text-align:center; border:2px solid {p1}; margin-bottom:8px;'><b>{res}</b><br>%{prob:.1f}</div>", unsafe_allow_html=True)

            st.divider()
            st.subheader("💡 6 KATMANLI STRATEJİ DANIŞMANI")
            s_c1, s_c2 = st.columns(2)
            s_c1.info(f"💎 **ULTRA-KASA:** {'1.5 ÜST' if u15 > 80 else '1X Çifte Şans' if evw+ber > 82 else 'Pas'}")
            s_c1.success(f"🟢 **GÜVENLİ:** {'Karşılıklı Gol' if kg > 60 else 'Maç Sonucu 1' if evw > 62 else 'Ev Gol'}")
            s_c1.warning(f"🟡 **ANA TERCİH:** {'2.5 ÜST' if u25 > 62 else 'Maç Sonucu 1'}")
            s_c2.warning(f"🟠 **DEĞERLİ ORAN:** {htft[0][0]} Senaryosu")
            s_c2.error(f"🔴 **BOMBACI (SÜRPRİZ):** {surp[0][0][0]}-{surp[0][0][1]} Skoru")
            s_c2.error(f"🔵 **CANLI:** {'70. Dakika Gol Kokusu' if u15 > 75 else 'Kart Bahsi'}")
            
            plt.figure(figsize=(6, 4), facecolor=p1)
            plt.text(0.5, 0.5, f"{ev_t} vs {dep_t}\nSkor: {int(np.round(fe))}-{int(np.round(fd))}\nKG: %{kg:.1f}", ha='center', color='white', weight='bold')
            plt.axis('off')
            buf = BytesIO(); plt.savefig(buf, format="png"); plt.close()
            st.download_button("🖼️ KUPON GÖRSELİNİ İNDİR", data=buf.getvalue(), file_name=f"{ev_t}_analiz.png", mime="image/png", use_container_width=True)

    with tab_banko:
        st.subheader("🔍 GLOBAL BANKO TARAYICI (0.8+ EŞİĞİ)")
        if st.button("🚀 GÜNÜN BANKOLARINI SÜZ"):
            with st.spinner("Tüm ligler taranıyor..."):
                found = []
                for n, c in LIGLER.items():
                    data_b = master_load_v51(c)
                    if data_b:
                        _, m_ev_b, m_dep_b, _, _, m_le, m_teams, _ = data_b
                        for i in range(min(5, len(m_teams)-1)):
                            t1, t2 = m_teams[i], m_teams[i+1]
                            fe_b, fd_b = m_ev_b.predict([[m_le.transform([t1])[0], m_le.transform([t2])[0]]])[0], m_dep_b.predict([[m_le.transform([t1])[0], m_le.transform([t2])[0]]])[0]
                            ew_b, _, _, u15_b, u25_b, kg_b, _, _, _ = engine_v51(fe_b, fd_b, 0, 0, 0.5, 0.5)
                            if any(p > 83 for p in [ew_b, u15_b, u25_b]):
                                found.append({"Lig": n, "Maç": f"{t1} - {t2}", "Tahmin": "1.5 ÜST" if u15_b > 83 else "MS 1", "Güven": f"%{max(ew_b, u15_b):.1f}"})
                st.table(found) if found else st.write("Şu an kriterlere uygun maç bulunamadı.")
 
