# PRO-SCOUT v57 GOD MODE - HATASIZ VE OPTİMİZE
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from scipy.stats import poisson
import matplotlib.pyplot as plt

st.set_page_config(page_title="PRO-SCOUT v57 GOD MODE", layout="wide")

# --- UI STYLING ---
st.markdown("""
<style>
body {background-color:#020617;}
.title {text-align:center;font-size:42px;color:white;font-weight:900;margin-bottom:20px;}
.card {background:#0f172a;padding:20px;border-radius:16px;text-align:center;color:white;margin-bottom:10px;}
.section {margin-top:20px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>⚽ PRO-SCOUT v57 GOD MODE</div>", unsafe_allow_html=True)

# --- LEAGUES ---
LEAGUES = {
    'TR Süper Lig': 'T1','TR 1.Lig': 'T2',
    'ENG Premier': 'E0','ENG Champ': 'E1','ENG L1': 'E2','ENG L2': 'E3',
    'ESP La Liga': 'SP1','ESP Segunda': 'SP2',
    'GER Bundesliga': 'D1','GER 2.Lig': 'D2',
    'ITA Serie A': 'I1','ITA Serie B': 'I2',
    'FRA Ligue 1': 'F1','FRA Ligue 2': 'F2'
}

# --- LOAD DATA ---
@st.cache_data(show_spinner=False)
def load_data(code):
    df = pd.DataFrame()
    for season in ["2324","2425","2526"]:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            df = pd.concat([df, pd.read_csv(url)], ignore_index=True)
        except:
            continue
    df = df.dropna(subset=['HomeTeam','AwayTeam','FTHG','FTAG'])
    
    # Form & Defense Features
    df['form_home'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['form_away'] = df.groupby('AwayTeam')['FTAG'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['def_home'] = df.groupby('HomeTeam')['FTAG'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['def_away'] = df.groupby('AwayTeam')['FTHG'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    df.fillna(0, inplace=True)
    
    le = LabelEncoder()
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    le.fit(teams)
    
    df['h'] = le.transform(df['HomeTeam'])
    df['a'] = le.transform(df['AwayTeam'])
    
    X = df[['h','a','form_home','form_away','def_home','def_away']]
    model_h = XGBRegressor(n_estimators=70, max_depth=4, random_state=42)
    model_a = XGBRegressor(n_estimators=70, max_depth=4, random_state=42)
    
    model_h.fit(X, df['FTHG'])
    model_a.fit(X, df['FTAG'])
    
    return df, model_h, model_a, le, teams

# --- PREDICT FUNCTION ---
def predict(home, away, df, mh, ma, le):
    h = le.transform([home])[0]
    a = le.transform([away])[0]
    
    last_row = df.iloc[-1]
    X_pred = [[h, a, last_row['form_home'], last_row['form_away'], last_row['def_home'], last_row['def_away']]]
    
    gh = max(mh.predict(X_pred)[0], 0.2)
    ga = max(ma.predict(X_pred)[0], 0.2)
    
    probs = np.outer([poisson.pmf(i, gh) for i in range(6)],
                     [poisson.pmf(j, ga) for j in range(6)])
    
    hw = np.sum(np.tril(probs, -1))*100
    dr = np.sum(np.diag(probs))*100
    aw = np.sum(np.triu(probs, 1))*100
    
    return gh, ga, hw, dr, aw

# --- TEAM RADAR ---
def team_strength(df, team):
    sub = df[df['HomeTeam']==team]
    vals = [sub['FTHG'].mean(), sub['FTAG'].mean(), sub['form_home'].mean(), sub['def_home'].mean()]
    return vals

# --- UI ---
league = st.selectbox("Lig Seçin", list(LEAGUES.keys()))
data = load_data(LEAGUES[league])

if data:
    df, mh, ma, le, teams = data
    tab1, tab2, tab3 = st.tabs(["Analiz","Radar","Banko AI"])
    
    # Analiz Tab
    with tab1:
        c1, c2 = st.columns(2)
        home = c1.selectbox("Ev Sahibi Takım", teams)
        away = c2.selectbox("Deplasman Takımı", teams)
        
        if st.button("Tahmin Et"):
            gh, ga, hw, dr, aw = predict(home, away, df, mh, ma, le)
            
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"<div class='card'>EV %{hw:.1f}</div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='card'>BERABER %{dr:.1f}</div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='card'>DEP %{aw:.1f}</div>", unsafe_allow_html=True)
            
            st.success(f"Tahmini Skor: {round(gh)} - {round(ga)}")
    
    # Radar Tab
    with tab2:
        team_sel = st.selectbox("Takım Seç", teams, key="radar_team")
        radar_vals = team_strength(df, team_sel)
        
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, polar=True)
        angles = np.linspace(0, 2*np.pi, len(radar_vals), endpoint=False)
        radar_vals = np.append(radar_vals, radar_vals[0])
        angles = np.append(angles, angles[0])
        
        ax.plot(angles, radar_vals, 'o-', linewidth=2)
        ax.fill(angles, radar_vals, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['Hücum','Savunma','Form','Defans'])
        st.pyplot(fig)
    
    # Banko AI Tab
    with tab3:
        if st.button("Günlük Banko Tarama"):
            for i in range(min(12, len(teams)-1)):
                gh, ga, hw, dr, aw = predict(teams[i], teams[i+1], df, mh, ma, le)
                if hw > 72 or aw > 72:
                    st.write(f"🔥 {teams[i]} vs {teams[i+1]} → %{max(hw,aw):.1f}")
