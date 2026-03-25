#PRO-SCOUT v54 PREMIUM

Ultra Professional Football Prediction Dashboard

import streamlit as st import pandas as pd import numpy as np from sklearn.preprocessing import LabelEncoder from xgboost import XGBRegressor from scipy.stats import poisson import matplotlib.pyplot as plt

--- PAGE CONFIG ---

st.set_page_config(page_title="PRO-SCOUT v54 PREMIUM", layout="wide")

--- CUSTOM CSS (PREMIUM UI) ---

st.markdown("""

<style>
body {background-color: #0f172a;}
.metric-card {
    background: linear-gradient(135deg,#1e293b,#334155);
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:white;
    box-shadow:0 10px 25px rgba(0,0,0,0.4);
}
.title {
    text-align:center;
    font-size:42px;
    font-weight:700;
    color:white;
}
.subtitle {
    text-align:center;
    color:#94a3b8;
    margin-bottom:30px;
}
</style>""", unsafe_allow_html=True)

--- HEADER ---

st.markdown("<div class='title'>⚽ PRO-SCOUT v54 PREMIUM</div>", unsafe_allow_html=True) st.markdown("<div class='subtitle'>AI Powered Match Analysis System</div>", unsafe_allow_html=True)

--- LEAGUES ---

LIGLER = { 'Türkiye Süper Lig': 'T1', 'İngiltere Premier Lig': 'E0', 'İspanya La Liga': 'SP1' }

--- DATA LOADER ---

@st.cache_data def load_data(code): df_all = pd.DataFrame() for s in ["2324","2425","2526"]: try: url = f"https://www.football-data.co.uk/mmz4281/{s}/{code}.csv" df = pd.read_csv(url) df_all = pd.concat([df_all, df]) except: continue

df_all['Date'] = pd.to_datetime(df_all['Date'], errors='coerce')
df_all = df_all.dropna()

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

model_h = XGBRegressor(n_estimators=200, max_depth=5).fit(X, df_all['FTHG'])
model_a = XGBRegressor(n_estimators=200, max_depth=5).fit(X, df_all['FTAG'])

return df_all, model_h, model_a, le, teams

--- ENGINE ---

def predict(home, away, df, mh, ma, le): h = le.transform([home])[0] a = le.transform([away])[0]

stats = df.iloc[-1]
X = [[h, a, stats['home_attack'], stats['home_def'], stats['away_attack'], stats['away_def']]]

gh = max(mh.predict(X)[0], 0.2)
ga = max(ma.predict(X)[0], 0.2)

matrix = np.outer([poisson.pmf(i, gh) for i in range(6)],
                  [poisson.pmf(j, ga) for j in range(6)])

hw = np.sum(np.tril(matrix, -1))*100
dr = np.sum(np.diag(matrix))*100
aw = np.sum(np.triu(matrix, 1))*100

return gh, ga, hw, dr, aw, matrix

--- UI ---

lig = st.selectbox("🌍 Lig Seç", list(LIGLER.keys())) data = load_data(LIGLER[lig])

if data: df, mh, ma, le, teams = data

col1, col2 = st.columns(2)
home = col1.selectbox("🏠 Ev Sahibi", teams)
away = col2.selectbox("🚀 Deplasman", teams)

if st.button("🚀 ANALİZİ BAŞLAT", use_container_width=True):
    gh, ga, hw, dr, aw, matrix = predict(home, away, df, mh, ma, le)

    st.markdown(f"## {home} vs {away}")

    # METRICS
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric-card'>🏠 EV<br><h2>%{hw:.1f}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'>🤝 BERABER<br><h2>%{dr:.1f}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'>🚀 DEP<br><h2>%{aw:.1f}</h2></div>", unsafe_allow_html=True)

    # SCORE
    st.success(f"🎯 Tahmini Skor: {round(gh)} - {round(ga)}")

    # CHART
    fig, ax = plt.subplots()
    ax.bar(["Home","Draw","Away"],[hw,dr,aw])
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

    # BETTING INSIGHT
    st.subheader("💡 AI Bahis Önerisi")

    if hw > 65:
        st.success("Maç Sonucu 1 Güçlü")
    elif aw > 65:
        st.success("Maç Sonucu 2 Güçlü")
    elif dr > 40:
        st.warning("Beraberlik Değerli")
    else:
        st.info("Dengeli Maç")
