import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import poisson
import os

st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor", page_icon="⚽", layout="wide"
)

st.markdown(
    """
<style>

.main {
    background-color: #0f172a;
}

.hero {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(90deg,#2563eb,#06b6d4);
    text-align: center;
    color: white;
    margin-bottom: 20px;
}

.hero h1 {
    font-size: 3rem;
    margin-bottom: 5px;
}

.hero p {
    font-size: 1.1rem;
}

.metric-box {
    background: #1e293b;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #334155;
}

.result-card {
    background: #1e293b;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
}

</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
    <h1> FIFA World Cup 2026 Predictor</h1>
    <p><b>Hybrid Football Prediction Model (Elo + Poisson)</b></p>
</div>
""",
    unsafe_allow_html=True,
)


# DATA LOADING
@st.cache_data
def load_data():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base_dir, "input", "results.csv")
    wc_path = os.path.join(base_dir, "input", "wc_26.csv")

    results = pd.read_csv(results_path)
    wc_26 = pd.read_csv(wc_path)

    wc_26 = wc_26.drop(columns=["Unnamed: 0"], errors="ignore")

    df = results.dropna(subset=["home_score"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ATTACK / DEFENSE
    modern = df[df["date"] >= "2000-01-01"]

    avg_goals = (modern["home_score"].sum() + modern["away_score"].sum()) / (
        len(modern) * 2
    )

    home = modern.groupby("home_team").agg(
        s=("home_score", "sum"), c=("away_score", "sum"), m=("home_score", "count")
    )

    away = modern.groupby("away_team").agg(
        s=("away_score", "sum"), c=("home_score", "sum"), m=("away_score", "count")
    )

    home.index.name = "team"
    away.index.name = "team"

    team = home.add(away, fill_value=0).reset_index()

    prior_m = 5
    prior_g = avg_goals * prior_m

    team["attack"] = ((team["s"] + prior_g) / (team["m"] + prior_m)) / avg_goals
    team["defense"] = ((team["c"] + prior_g) / (team["m"] + prior_m)) / avg_goals

    attack_map = dict(zip(team["team"], team["attack"]))
    defense_map = dict(zip(team["team"], team["defense"]))

    # ELO SYSTEM
    elo = {}

    def get_elo(t):
        if t not in elo:
            elo[t] = 1500
        return elo[t]

    def k_factor(t):
        t = str(t).lower()
        if "world cup" in t:
            return 60
        if "qualification" in t:
            return 40
        return 20

    home_elo, away_elo = [], []

    for _, r in df.iterrows():
        h, a = r["home_team"], r["away_team"]

        he, ae = get_elo(h), get_elo(a)

        home_elo.append(he)
        away_elo.append(ae)

        home_adv = 100 if not r["neutral"] else 0

        diff = (he + home_adv) - ae
        exp = 1 / (10 ** (-diff / 400) + 1)

        if r["home_score"] > r["away_score"]:
            res_h = 1
        elif r["home_score"] < r["away_score"]:
            res_h = 0
        else:
            res_h = 0.5

        k = k_factor(r["tournament"])

        elo[h] = he + k * (res_h - exp)
        elo[a] = ae + k * ((1 - res_h) - (1 - exp))

    df["home_elo"] = home_elo
    df["away_elo"] = away_elo
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    # MODEL FEATURES
    modern = df[df["date"] >= "2000-01-01"].copy()

    modern["home_att"] = modern["home_team"].map(attack_map).fillna(1)
    modern["away_att"] = modern["away_team"].map(attack_map).fillna(1)
    modern["home_def"] = modern["home_team"].map(defense_map).fillna(1)
    modern["away_def"] = modern["away_team"].map(defense_map).fillna(1)

    for col in ["elo_diff", "home_att", "away_att", "home_def", "away_def"]:
        modern[col] = (modern[col] - modern[col].mean()) / (modern[col].std() + 1e-6)

    # POISSON MODELS
    home_model = smf.poisson(
        "home_score ~ elo_diff + home_att + away_def", data=modern
    ).fit(disp=0)

    away_model = smf.poisson(
        "away_score ~ elo_diff + away_att + home_def", data=modern
    ).fit(disp=0)

    return elo, attack_map, defense_map, home_model.params, away_model.params, wc_26


elo, att, deff, h_p, a_p, wc = load_data()

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        """
    <div class="metric-box">
        <h2>48</h2>
        <p>Teams</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
    <div class="metric-box">
        <h2>104</h2>
        <p>Matches</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
    <div class="metric-box">
        <h2>ELO</h2>
        <p>Rating System</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        """
    <div class="metric-box">
        <h2>Poisson</h2>
        <p>Goal Model</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.divider()
