# PRO-SCOUT v54 PREMIUM
# AI Football Prediction Dashboard

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from scipy.stats import poisson
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="PRO-SCOUT v54", layout="wide")

# --- CUSTOM UI ---
st.markdown("""
<style>
body {background-color: #0f172a;}
.title {
    text-align:center;
    font-size:40px;
    color:white;
    font-weight:700;
}
.card {
    background: linear-gradient(135deg,#1e293b,#334155);
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:white;
    box-shadow:0 8px 20px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>⚽ PRO-SCOUT v54 PREMIUM</div>", unsafe_allow_html=True)

# --- LEAGUES ---
LIGLER = {
    'Türkiye Süper Lig': 'T1',
    'İngiltere Premier Lig': 'E0',
    'İspanya La Liga': 'SP1'
}

# --- DATA LOAD ---
@st.cache_data
def load_data(code):
    df_all = pd.DataFrame()

    for s in ["2324","2425","2526"]:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{s}/{code}.csv"
            df = pd.read_csv(url)
            df_all = pd.concat([df_all, df])
        except:
            continue

    df_all = df_all.dropna(subset=['HomeTeam','AwayTeam','FTHG','FTAG'])

    # FEATURES
    df_all['home_attack'] = df_all.groupby('HomeTeam')['FTHG'].transform('mean')
    df_all['home_def'] = df_all.groupby('HomeTeam')['FTAG'].transform('mean')
    df_all['away_attack'] = df_all.groupby('AwayTeam')['FTAG'].transform('mean')
    df_all['away_def'] = df_all.groupby('AwayTeam')['FTHG'].transform('mean')

    le = LabelEncoder()
    teams = pd.concat([df_all['HomeTeam'], df_all['AwayTeam']]).unique()
    le.fit(teams)

    df_all['h'] = le.transform(df_all['HomeTeam'])
    df_all['a'] = le.transform(df_all['AwayTeam'])

    X = df_all[['h','a','home_attack','home_def','away_attack','away_def']]

    model_h = XGBRegressor(n_estimators=150, max_depth=5)
    model_a = XGBRegressor(n_estimators=150, max_depth=5)

    model_h.fit(X, df_all['FTHG'])
    model_a.fit(X, df_all['FTAG'])

    return df_all, model_h, model_a, le, teams

# --- PREDICTION ENGINE ---
def predict(home, away, df, mh, ma, le):
    h = le.transform([home])[0]
    a = le.transform([away])[0]

    stats = df.iloc[-1]

    X = [[h, a, stats['home_attack'], stats['home_def'],
          stats['away_attack'], stats['away_def']]]

    gh = max(mh.predict(X)[0], 0.2)
    ga = max(ma.predict(X)[0], 0.2)

    matrix = np.outer(
        [poisson.pmf(i, gh) for i in range(6)],
        [poisson.pmf(j, ga) for j in range(6)]
    )

    home_win = np.sum(np.tril(matrix, -1))*100
    draw = np.sum(np.diag(matrix))*100
    away_win = np.sum(np.triu(matrix, 1))*100

    return gh, ga, home_win, draw, away_win, matrix

# --- UI ---
lig = st.selectbox("🌍 Lig Seç", list(LIGLER.keys()))
data = load_data(LIGLER[lig])

if data:
    df, mh, ma, le, teams = data

    col1, col2 = st.columns(2)
    home = col1.selectbox("🏠 Ev Sahibi", teams)
    away = col2.selectbox("🚀 Deplasman", teams)

    if st.button("🚀 ANALİZİ BAŞLAT", use_container_width=True):

        gh, ga, hw, dr, aw, matrix = predict(home, away, df, mh, ma, le)

        st.markdown(f"## {home} vs {away}")

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='card'>🏠 EV<br><h2>%{hw:.1f}</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card'>🤝 BERABER<br><h2>%{dr:.1f}</h2></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='card'>🚀 DEP<br><h2>%{aw:.1f}</h2></div>", unsafe_allow_html=True)

        st.success(f"🎯 Tahmini Skor: {round(gh)} - {round(ga)}")

        # Chart
        fig, ax = plt.subplots()
        ax.bar(["Home","Draw","Away"], [hw, dr, aw])
        st.pyplot(fig)

        # TOP SCORES
        st.subheader("🔥 En Olası Skorlar")

        scores = []
        for i in range(5):
            for j in range(5):
                scores.append(((i,j), matrix[i,j]*100))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:6]

        cols = st.columns(3)
        for i, (sc, pr) in enumerate(scores):
            cols[i%3].info(f"{sc[0]} - {sc[1]}  (%{pr:.1f})")

        # AI YORUM
        st.subheader("💡 AI Yorum")

        if hw > 65:
            st.success("Güçlü: Ev Sahibi Kazanır")
        elif aw > 65:
            st.success("Güçlü: Deplasman Kazanır")
        elif dr > 40:
            st.warning("Beraberlik Değerli")
        else:
            st.info("Dengeli Maç")
