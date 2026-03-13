import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib
import os

# ── Page config ─────────────────────────────────────
st.set_page_config(
    page_title="Cricket Win Predictor",
    page_icon="🏏",
    layout="centered"
)

# ── Load model and scaler (cached so they load only once) ───
@st.cache_resource
def load_model_and_scaler():
    model  = tf.keras.models.load_model("models/cricket_model.keras")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

FEATURE_COLS = [
    "balls_remaining", "runs_remaining", "wickets_remaining", "runs_scored",
    "current_run_rate", "required_run_rate", "target",
    "run_rate_diff", "balls_remaining_ratio", "wickets_remaining_ratio",
    "runs_per_ball_required", "match_completion", "pressure_index",
    "chase_difficulty", "recent_form", "win_momentum", "danger_zone",
]

SEQUENCE_LEN = 10


def compute_features(runs_scored, target, balls_bowled, wickets_fallen, match_type, recent_runs_input):
    """Given raw match state, compute all 17 features."""
    max_balls      = 120 if match_type == "T20" else 300
    balls_remaining   = max(max_balls - balls_bowled, 0)
    runs_remaining    = max(target - runs_scored, 0)
    wickets_remaining = max(10 - wickets_fallen, 0)

    current_rr  = (runs_scored / balls_bowled * 6) if balls_bowled > 0 else 0.0
    required_rr = (runs_remaining / balls_remaining * 6) if balls_remaining > 0 else 999.0

    run_rate_diff            = current_rr - required_rr
    balls_remaining_ratio    = balls_remaining / max_balls
    wickets_remaining_ratio  = wickets_remaining / 10.0
    runs_per_ball_required   = runs_remaining / max(balls_remaining, 1)
    match_completion         = balls_bowled / max_balls
    pressure_index           = (
        (runs_remaining / max(target, 1)) *
        (1.0 / (wickets_remaining + 1)) *
        match_completion
    )
    chase_difficulty = min(required_rr / 8.0, 5.0)
    recent_form      = min(recent_runs_input / 24.0, 1.0)
    win_momentum     = 1.0 if run_rate_diff > 0 else 0.0
    danger_zone      = 1.0 if (required_rr > 12 and wickets_remaining <= 5) else 0.0

    return [
        balls_remaining, runs_remaining, wickets_remaining, runs_scored,
        current_rr, required_rr, target,
        run_rate_diff, balls_remaining_ratio, wickets_remaining_ratio,
        runs_per_ball_required, match_completion, pressure_index,
        chase_difficulty, recent_form, win_momentum, danger_zone,
    ]


def predict_win_probability(ball_history):
    """
    ball_history: list of feature rows (each row = 17 features, unscaled)
    Takes last SEQUENCE_LEN rows, scales them, runs LSTM prediction.
    """
    if len(ball_history) < SEQUENCE_LEN:
        return None

    seq = np.array(ball_history[-SEQUENCE_LEN:], dtype=np.float32)
    seq_scaled = scaler.transform(seq)
    X = seq_scaled.reshape(1, SEQUENCE_LEN, len(FEATURE_COLS))
    prob = model.predict(X, verbose=0)[0][0]
    return float(prob)


# ── UI ──────────────────────────────────────────────
st.title("🏏 Cricket Win Predictor")
st.markdown("Enter the current match situation ball by ball to see live win probability.")

# ── Match setup (shown only once at start) ──────────
with st.expander("⚙️ Match Setup", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        match_type    = st.selectbox("Match Type", ["T20", "ODI"])
        batting_team  = st.text_input("Batting Team", "Team A")
    with col2:
        bowling_team  = st.text_input("Bowling Team", "Team B")
        target        = st.number_input("Target (runs to win)", min_value=1, max_value=500, value=180)

# ── Current ball input ───────────────────────────────
st.markdown("---")
st.subheader("📊 Current Match State")

col1, col2, col3 = st.columns(3)
with col1:
    runs_scored    = st.number_input("Runs Scored", min_value=0, max_value=500, value=0)
    wickets_fallen = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=0)
with col2:
    balls_bowled   = st.number_input("Balls Bowled", min_value=0, max_value=300, value=0)
    recent_runs    = st.number_input("Runs in Last 12 Balls", min_value=0, max_value=72, value=0)
with col3:
    max_balls = 120 if match_type == "T20" else 300
    balls_rem = max(max_balls - balls_bowled, 0)
    runs_rem  = max(target - runs_scored, 0)
    st.metric("Balls Remaining", balls_rem)
    st.metric("Runs Required",   runs_rem)

# ── Session state: store ball history ───────────────
if "ball_history" not in st.session_state:
    st.session_state.ball_history = []
if "prob_history" not in st.session_state:
    st.session_state.prob_history = []

col_add, col_reset = st.columns([2, 1])

with col_add:
    if st.button("➕ Add Ball & Predict", use_container_width=True):
        features = compute_features(
            runs_scored, target, balls_bowled, wickets_fallen, match_type, recent_runs
        )
        st.session_state.ball_history.append(features)

        prob = predict_win_probability(st.session_state.ball_history)

        if prob is not None:
            st.session_state.prob_history.append({
                "ball": balls_bowled,
                "probability": round(prob * 100, 1)
            })

with col_reset:
    if st.button("🔄 Reset Match", use_container_width=True):
        st.session_state.ball_history = []
        st.session_state.prob_history = []
        st.rerun()

# ── Show current win probability ─────────────────────
st.markdown("---")

if st.session_state.prob_history:
    latest = st.session_state.prob_history[-1]["probability"]

    # Color based on probability
    if latest >= 65:
        color = "#2ecc71"   # green  — batting team winning
        status = "🟢 Likely Win"
    elif latest <= 35:
        color = "#e74c3c"   # red    — batting team losing
        status = "🔴 Likely Loss"
    else:
        color = "#f39c12"   # orange — close match
        status = "🟡 Close Match"

    st.markdown(
        f"""
        <div style='text-align:center; padding:20px; border-radius:12px; background:{color}22; border: 2px solid {color}'>
            <h2 style='color:{color}; margin:0'>{batting_team} Win Probability</h2>
            <h1 style='color:{color}; font-size:64px; margin:8px 0'>{latest}%</h1>
            <p style='color:{color}; font-size:18px; margin:0'>{status}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Win probability chart
    if len(st.session_state.prob_history) > 1:
        st.markdown("### 📈 Win Probability Over Match")
        chart_df = pd.DataFrame(st.session_state.prob_history)
        chart_df = chart_df.rename(columns={"ball": "Ball", "probability": "Win Probability (%)"})
        chart_df = chart_df.set_index("Ball")
        st.line_chart(chart_df)

    # Match state summary table
    st.markdown("### 📋 Ball History")
    history_df = pd.DataFrame(st.session_state.prob_history)
    history_df.columns = ["Ball #", "Win Probability (%)"]
    st.dataframe(history_df, use_container_width=True, hide_index=True)

else:
    st.info(f"Add at least {SEQUENCE_LEN} balls to see win probability prediction.")
    st.markdown(f"*{SEQUENCE_LEN - len(st.session_state.ball_history)} more balls needed before first prediction.*")