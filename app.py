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
