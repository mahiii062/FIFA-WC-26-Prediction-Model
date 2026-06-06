import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import poisson

# PATH SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

results_path = os.path.join(BASE_DIR, 'input', 'results.csv')
wc_26_path = os.path.join(BASE_DIR, 'input', 'wc_26.csv')

results = pd.read_csv(results_path)
wc_26 = pd.read_csv(wc_26_path)

wc_26 = wc_26.drop(columns=['Unnamed: 0'], errors='ignore')

# TRAIN DATA
train_df = results.dropna(subset=['home_score']).copy()
train_df['date'] = pd.to_datetime(train_df['date'])
train_df = train_df.sort_values('date').reset_index(drop=True)

HOST_NATIONS = ['United States', 'Mexico', 'Canada']

# ATTACK / DEFENSE
modern = train_df[train_df['date'] >= '2000-01-01']

global_avg = (
    (modern['home_score'].sum() + modern['away_score'].sum())
    / (len(modern) * 2)
)

home = modern.groupby('home_team').agg(
    s=('home_score', 'sum'),
    c=('away_score', 'sum'),
    m=('home_score', 'count')
)

away = modern.groupby('away_team').agg(
    s=('away_score', 'sum'),
    c=('home_score', 'sum'),
    m=('away_score', 'count')
)

home.index.name = "team"
away.index.name = "team"

team = home.add(away, fill_value=0).reset_index()

prior_m = 5
prior_g = global_avg * prior_m

team["attack"] = (
    (team["s"] + prior_g) / (team["m"] + prior_m)
) / global_avg

team["defense"] = (
    (team["c"] + prior_g) / (team["m"] + prior_m)
) / global_avg

att_map = dict(zip(team["team"], team["attack"]))
def_map = dict(zip(team["team"], team["defense"]))

# ELO SYSTEM
elo = {}

def get_elo(t):
    elo.setdefault(t, 1500)
    return elo[t]

def k_factor(t):
    t = str(t).lower()
    if "world cup" in t:
        return 60
    if "qualification" in t:
        return 40
    return 20

home_elos, away_elos = [], []
max_date = train_df["date"].max()

for _, r in train_df.iterrows():
    h, a = r["home_team"], r["away_team"]

    he, ae = get_elo(h), get_elo(a)

    home_elos.append(he)
    away_elos.append(ae)

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

train_df["home_elo"] = home_elos
train_df["away_elo"] = away_elos

train_df["elo_diff"] = train_df["home_elo"] - train_df["away_elo"]

# FEATURES
modern_df = train_df[train_df["date"] >= "2000-01-01"].copy()

modern_df["home_att"] = modern_df["home_team"].map(att_map).fillna(1)
modern_df["away_att"] = modern_df["away_team"].map(att_map).fillna(1)
modern_df["home_def"] = modern_df["home_team"].map(def_map).fillna(1)
modern_df["away_def"] = modern_df["away_team"].map(def_map).fillna(1)

# True Z-score standardization applied at train-time to match load_data()
for c in ["elo_diff", "home_att", "away_att", "home_def", "away_def"]:
    modern_df[c] = (modern_df[c] - modern_df[c].mean()) / modern_df[c].std()

# POISSON MODELS
home_model = smf.poisson(
    "home_score ~ elo_diff + home_att + away_def",
    data=modern_df
).fit(disp=0)

away_model = smf.poisson(
    "away_score ~ elo_diff + away_att + home_def",
    data=modern_df
).fit(disp=0)


# PREDICTION
def predict_match(home, away, neutral, country):
    he = elo.get(home, 1500)
    ae = elo.get(away, 1500)

    h_adv = 100 if not neutral else 0
    diff = he - ae + h_adv

    h_att = att_map.get(home, 1)
    a_att = att_map.get(away, 1)
    h_def = def_map.get(home, 1)
    a_def = def_map.get(away, 1)

    # Constant scale translation
    def norm(x, mean=0, std=200):
        return (x - mean) / std

    diff = norm(diff)
    # Note: h_att, a_att, h_def, and a_def

    lam_h = np.exp(
        home_model.params["Intercept"]
        + home_model.params["elo_diff"] * diff
        + home_model.params["home_att"] * h_att
        + home_model.params["away_def"] * a_def
    )

    lam_a = np.exp(
        away_model.params["Intercept"]
        + away_model.params["elo_diff"] * diff
        + away_model.params["away_att"] * a_att
        + away_model.params["home_def"] * h_def
    )

    lam_h = np.clip(lam_h, 0.05, 5)
    lam_a = np.clip(lam_a, 0.05, 5)

    max_goals = 7

    p_h = poisson.pmf(np.arange(max_goals+1), lam_h)
    p_a = poisson.pmf(np.arange(max_goals+1), lam_a)

    mat = np.outer(p_h, p_a)

    ph = np.sum(np.tril(mat, -1))
    pd = np.sum(np.diag(mat))
    pa = np.sum(np.triu(mat, 1))

    total = ph + pd + pa

    ph, pd, pa = ph/total, pd/total, pa/total

    return {
        "home": home,
        "away": away,
        "home_win": ph,
        "draw": pd,
        "away_win": pa,
        "score": f"{np.argmax(p_h)}-{np.argmax(p_a)}"
    }


# SAVE WC PREDICTIONS
out = []

for _, r in wc_26.iterrows():
    res = predict_match(
        r["home_team"],
        r["away_team"],
        r["neutral"],
        r["country"]
    )
    out.append(res)

df_out = pd.DataFrame(out)

output_dir = os.path.join(BASE_DIR, "output")
os.makedirs(output_dir, exist_ok=True)

df_out.to_csv(os.path.join(output_dir, "wc_2026_final.csv"), index=False)

print("Saved predictions to output/wc_2026_final.csv")