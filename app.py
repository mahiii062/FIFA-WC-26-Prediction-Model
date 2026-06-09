import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import poisson
import os

st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor", page_icon="⚽", layout="wide"
)

# st.title("FIFA World Cup 2026")
# st.write("Hybrid Football Prediction Model (Elo + Poisson)")

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
    <h1>FIFA World Cup 2026 Predictor</h1>
    <p>Hybrid Football Prediction Model (Elo + Poisson)</p>
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


# MATCH PREDICTOR
def predict(home, away, neutral):

    he = elo.get(home, 1500)
    ae = elo.get(away, 1500)

    h_adv = 100 if not neutral else 0
    diff = he - ae + h_adv

    ha, aa = att.get(home, 1), att.get(away, 1)
    hd, ad = deff.get(home, 1), deff.get(away, 1)

    diff = diff / 200  # normalization

    lam_h = np.exp(
        h_p["Intercept"]
        + h_p["elo_diff"] * diff
        + h_p["home_att"] * ha
        + h_p["away_def"] * ad
    )

    lam_a = np.exp(
        a_p["Intercept"]
        + a_p["elo_diff"] * diff
        + a_p["away_att"] * aa
        + a_p["home_def"] * hd
    )

    lam_h = np.clip(lam_h, 0.05, 5)
    lam_a = np.clip(lam_a, 0.05, 5)

    max_g = 7

    ph = poisson.pmf(np.arange(max_g + 1), lam_h)
    pa = poisson.pmf(np.arange(max_g + 1), lam_a)

    mat = np.outer(ph, pa)

    p_home = np.sum(np.tril(mat, -1))
    p_draw = np.sum(np.diag(mat))
    p_away = np.sum(np.triu(mat, 1))

    total = p_home + p_draw + p_away + 1e-9

    p_home /= total
    p_draw /= total
    p_away /= total

    score = f"{np.argmax(ph)}-{np.argmax(pa)}"

    if np.argmax(ph) > np.argmax(pa):
        verdict = "Home Win"
    elif np.argmax(pa) > np.argmax(ph):
        verdict = "Away Win"
    else:
        verdict = "Draw"

    return p_home, p_draw, p_away, score, verdict


# UI
tab1, tab2 = st.tabs(["Match Predictions for Group Stage", "Match Simulator"])

# TAB 1
with tab1:
    st.subheader("Fixtures Prediction")

    preds = []

    for _, row in wc.iterrows():
        p1, pd_, p2, score, verdict = predict(
            row["home_team"], row["away_team"], row["neutral"]
        )

        preds.append(
            {
                "Home": row["home_team"],
                "Away": row["away_team"],
                "Home Win %": f"{p1*100:.1f}%",
                "Draw %": f"{pd_*100:.1f}%",
                "Away Win %": f"{p2*100:.1f}%",
                "Score": score,
                "Result": verdict,
            }
        )

    pred_df = pd.DataFrame(preds)

    st.dataframe(pred_df, use_container_width=True, hide_index=True, height=650)


# TAB 2
with tab2:
    st.subheader("Match Simulator")

    teams = sorted(list(elo.keys()))

    c1, c2 = st.columns(2)

    h = c1.selectbox("Home", teams)
    a = c2.selectbox("Away", teams)

    st.markdown("### Select Teams")

if st.button("Predict Match", width="stretch"):

    p1, pd_, p2, score, verdict = predict(h, a, True)

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            label=f"{h}",
            value=f"{p1*100:.1f}%"
        )

    with m2:
        st.metric(
            label="Draw",
            value=f"{pd_*100:.1f}%"
        )

    with m3:
        st.metric(
            label=f"{a}",
            value=f"{p2*100:.1f}%"
        )

    st.success(f"Predicted Score: {score}")

    st.markdown(
        f"Expected Result: **{verdict}**"
    )