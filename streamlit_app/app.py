"""
Viral Video Predictor - Streamlit front end.
Run from the project root:   streamlit run streamlit_app/app.py
Needs api/model.json (created by:  python train_export.py USvideos.csv).
"""
import json
import math
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Viral Video Predictor", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------------- styling
st.markdown(
    """
<style>
  #MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}
  .block-container {padding-top: 2rem; max-width: 1200px;}
  .hero {
      padding: 2rem 2.2rem; border-radius: 18px; margin-bottom: 1.4rem;
      background: linear-gradient(135deg, #1a2238 0%, #0f1526 60%, #2a1218 100%);
      border: 1px solid rgba(255,255,255,0.07);
  }
  .hero h1 {margin: 0; font-size: 2.1rem; letter-spacing: .3px; color: #F4F6FB;}
  .hero p {margin: .5rem 0 0; color: #9AA4BC; font-size: 1.02rem;}
  .hero .tag {
      display: inline-block; font-size: .72rem; letter-spacing: 1.6px; text-transform: uppercase;
      color: #E5484D; margin-bottom: .6rem; font-weight: 600;
  }
  .card {
      background: #131A2A; border: 1px solid rgba(255,255,255,0.07); border-radius: 14px;
      padding: 1.1rem 1.3rem; height: 100%;
  }
  .card .label {color: #8C97B2; font-size: .78rem; text-transform: uppercase; letter-spacing: 1.2px;}
  .card .value {color: #F4F6FB; font-size: 1.9rem; font-weight: 700; margin-top: .25rem; line-height: 1.15;}
  .card .sub {color: #8C97B2; font-size: .85rem; margin-top: .3rem;}
  .good {color: #3DD68C !important;} .bad {color: #E5484D !important;}
  .section {font-size: 1.05rem; font-weight: 600; color: #E6E9F0; margin: 1.4rem 0 .4rem;}
  [data-testid="stSidebar"] {border-right: 1px solid rgba(255,255,255,0.06);}
  .stTabs [data-baseweb="tab-list"] {gap: 1.2rem;}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- model
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [os.path.join(HERE, "model.json"), os.path.join(HERE, "..", "api", "model.json")]

LABELS = {
    "likes": "Likes",
    "dislikes": "Dislikes",
    "comment_count": "Comments",
    "title_length": "Title length",
    "days_since_publish": "Days since publish",
}


@st.cache_resource
def load_model():
    for path in CANDIDATES:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def predict(model, values):
    z = [(v - m) / s for v, m, s in zip(values, model["mean"], model["scale"])]
    views = model["linear"]["intercept"] + sum(c * x for c, x in zip(model["linear"]["coef"], z))
    contrib = [c * x for c, x in zip(model["logistic"]["coef"], z)]
    logit = model["logistic"]["intercept"] + sum(contrib)
    prob = 1.0 / (1.0 + math.exp(-max(min(logit, 500), -500)))
    return max(0, round(views)), prob, contrib


def card(label, value, sub="", cls=""):
    return f'<div class="card"><div class="label">{label}</div><div class="value {cls}">{value}</div><div class="sub">{sub}</div></div>'


model = load_model()
if model is None:
    st.error("model.json not found. Run `python train_export.py USvideos.csv` first, then restart the app.")
    st.stop()

features = model["features"]

# ----------------------------------------------------------------------------- sidebar inputs
PRESETS = {
    "Typical trending video": dict(likes=15000, dislikes=500, comment_count=1600, title_length=50, days_since_publish=2),
    "Rising creator": dict(likes=5000, dislikes=150, comment_count=500, title_length=45, days_since_publish=3),
    "Blockbuster": dict(likes=400000, dislikes=8000, comment_count=40000, title_length=40, days_since_publish=1),
}
for k, v in PRESETS["Typical trending video"].items():
    st.session_state.setdefault(k, v)


def apply_preset():
    for k, v in PRESETS[st.session_state["preset"]].items():
        st.session_state[k] = v


with st.sidebar:
    st.markdown("### Video details")
    st.selectbox("Quick start", list(PRESETS), key="preset", on_change=apply_preset)
    st.markdown("---")
    st.number_input("Likes", min_value=0, step=500, key="likes")
    st.number_input("Dislikes", min_value=0, step=50, key="dislikes")
    st.number_input("Comments", min_value=0, step=100, key="comment_count")
    st.number_input("Title length (characters)", min_value=0, max_value=200, step=1, key="title_length")
    st.number_input("Days since publish", min_value=0, max_value=3650, step=1, key="days_since_publish")
    st.caption("Change any value and the prediction updates instantly.")

values = [float(st.session_state[f]) for f in features]
pred_views, prob, contrib = predict(model, values)
is_viral = prob >= 0.5
threshold = model["viral_threshold"]

# ----------------------------------------------------------------------------- header
st.markdown(
    """
<div class="hero">
  <div class="tag">Machine learning · YouTube analytics</div>
  <h1>Viral Video Predictor</h1>
  <p>Estimate a video's view count and its chance of landing in the top 25% of trending videos.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab_pred, tab_model, tab_about = st.tabs(["Prediction", "Model insights", "About"])

# ----------------------------------------------------------------------------- prediction tab
with tab_pred:
    c1, c2, c3 = st.columns(3)
    c1.markdown(card("Predicted views", f"{pred_views:,}", "Linear regression estimate"), unsafe_allow_html=True)
    c2.markdown(card("Viral probability", f"{prob:.1%}", "Logistic regression"), unsafe_allow_html=True)
    c3.markdown(
        card(
            "Verdict",
            "Likely viral" if is_viral else "Not likely viral",
            f"Viral = more than {threshold:,.0f} views",
            "good" if is_viral else "bad",
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.25])

    with left:
        st.markdown('<div class="section">Virality gauge</div>', unsafe_allow_html=True)
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 44, "color": "#F4F6FB"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8C97B2"},
                    "bar": {"color": "#3DD68C" if is_viral else "#E5484D", "thickness": 0.28},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 33], "color": "rgba(229,72,77,0.18)"},
                        {"range": [33, 66], "color": "rgba(240,180,41,0.18)"},
                        {"range": [66, 100], "color": "rgba(61,214,140,0.18)"},
                    ],
                    "threshold": {"line": {"color": "#F4F6FB", "width": 3}, "thickness": 0.8, "value": 50},
                },
            )
        )
        gauge.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E6E9F0"})
        st.plotly_chart(gauge)

    with right:
        st.markdown('<div class="section">What is driving this prediction?</div>', unsafe_allow_html=True)
        names = [LABELS[f] for f in features]
        order = sorted(range(len(contrib)), key=lambda i: abs(contrib[i]))
        bars = go.Figure(
            go.Bar(
                x=[contrib[i] for i in order],
                y=[names[i] for i in order],
                orientation="h",
                marker_color=["#3DD68C" if contrib[i] >= 0 else "#E5484D" for i in order],
                hovertemplate="%{y}: %{x:+.2f}<extra></extra>",
            )
        )
        bars.update_layout(
            height=300, margin=dict(l=10, r=20, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E6E9F0"}, xaxis_title="Push on log-odds of going viral",
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=True, zerolinecolor="rgba(255,255,255,0.4)"),
        )
        st.plotly_chart(bars)

    st.caption("Green bars push the video toward viral, red bars push it away. "
               "Values are relative to an average trending video.")

# ----------------------------------------------------------------------------- model tab
with tab_model:
    m = model["metrics"]
    st.markdown('<div class="section">Test-set performance</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].markdown(card("R² (views)", f"{m['r2']:.3f}", "Regression"), unsafe_allow_html=True)
    cols[1].markdown(card("MAE (views)", f"{m['mae']:,.0f}", "Average error"), unsafe_allow_html=True)
    cols[2].markdown(card("Accuracy", f"{m['accuracy']:.1%}", "Classification"), unsafe_allow_html=True)
    cols[3].markdown(card("ROC-AUC", f"{m['roc_auc']:.3f}", "Classification"), unsafe_allow_html=True)
    st.write("")
    cols = st.columns(3)
    cols[0].markdown(card("Precision", f"{m['precision']:.1%}"), unsafe_allow_html=True)
    cols[1].markdown(card("Recall", f"{m['recall']:.1%}"), unsafe_allow_html=True)
    cols[2].markdown(card("F1-score", f"{m['f1']:.3f}"), unsafe_allow_html=True)

    st.markdown('<div class="section">Feature importance (standardised coefficients)</div>', unsafe_allow_html=True)
    coef_df = pd.DataFrame(
        {
            "Feature": [LABELS[f] for f in features],
            "Linear regression (views)": model["linear"]["coef"],
            "Logistic regression (viral)": model["logistic"]["coef"],
        }
    )
    fig = go.Figure()
    fig.add_bar(name="Logistic (viral)", x=coef_df["Feature"], y=coef_df["Logistic regression (viral)"],
                marker_color="#E5484D", yaxis="y2", offsetgroup=2)
    fig.add_bar(name="Linear (views)", x=coef_df["Feature"], y=coef_df["Linear regression (views)"],
                marker_color="#5B8DEF", offsetgroup=1)
    fig.update_layout(
        height=380, barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E6E9F0"}, margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.12),
        yaxis=dict(title="Linear coefficient", gridcolor="rgba(255,255,255,0.08)"),
        yaxis2=dict(title="Logistic coefficient", overlaying="y", side="right", showgrid=False),
    )
    st.plotly_chart(fig)
    st.dataframe(coef_df.round(3), hide_index=True)
    st.caption(f"Trained on {model['trained_on_rows']:,} trending videos. "
               f"Viral threshold (75th percentile of views): {threshold:,.0f}.")

# ----------------------------------------------------------------------------- about tab
with tab_about:
    st.markdown(
        f"""
#### How it works
Two models are trained on the US trending-videos dataset:

- **Linear regression** estimates the number of views.
- **Logistic regression** estimates the probability that a video is *viral*, defined as having
  more than **{threshold:,.0f}** views (the top 25%).

Inputs are standardised with a `StandardScaler` before they reach the models. The five inputs are
likes, dislikes, comment count, title length and days between publishing and trending.

#### Read the results carefully
- Likes, dislikes and comments are partly a *result* of high views, so this shows association, not cause.
- Predictions are most reliable for values within the range of the training data; extreme inputs extrapolate.
- The model ignores channel, topic and thumbnail quality, all of which matter in reality.
"""
    )
