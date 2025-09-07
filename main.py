# main.py  —  Neon/Cyberpunk EEG Predictor (no files, only sliders)
import time
import numpy as np
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Neon Hyperdrive EEG", layout="wide")

# ----------------------- THEME / CSS -----------------------
st.markdown("""
<style>
:root{
  --bg:#0b0f14; --fg:#e6f7ff;
  --cyan:#00FFE1; --mag:#A100FF; --pink:#FF007A; --sky:#00C2FF; --lime:#8aff7a;
}
html,body,[data-testid="stApp"]{
  background: radial-gradient(1200px 800px at 10% 10%, rgba(18,30,48,.6), transparent 50%),
              radial-gradient(1200px 800px at 90% 10%, rgba(48,18,30,.5), transparent 50%),
              linear-gradient(135deg,#0b0f14 20%, #0e1320 50%, #05070c 100%);
  color: var(--fg);
}
h1,h2,h3,h4{
  text-shadow: 0 0 12px rgba(161,0,255,.5), 0 0 24px rgba(0,194,255,.3);
  letter-spacing:.5px;
}
.neon-title{
  font-size: clamp(28px, 5vw, 42px);
  font-weight: 800;
  background: linear-gradient(90deg,var(--cyan),var(--mag),var(--pink));
  -webkit-background-clip: text; background-clip: text; color: transparent;
  filter: drop-shadow(0 0 8px rgba(0,255,225,.35));
}
.sub{
  color:#9cc; opacity:.8; margin-top:-10px; font-size:.9rem;
}
.grid{ display:grid; gap:22px; grid-template-columns: repeat(12, 1fr); }
.card{
  grid-column: span 12;
  background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 18px;
  padding: 18px 18px 26px;
  backdrop-filter: blur(10px);
  position: relative;
  overflow:hidden;
  box-shadow: inset 0 0 30px rgba(161,0,255,.05), 0 20px 60px rgba(0,0,0,.35);
}
@media(min-width:992px){
  .left{ grid-column: span 6; }
  .right{ grid-column: span 6; }
}
.chip-edges:before,.chip-edges:after{
  content:""; position:absolute; inset:-1px;
  background:
    repeating-linear-gradient(90deg, transparent 0 16px, rgba(0,255,225,.15) 16px 18px),
    repeating-linear-gradient(0deg,  transparent 0 16px, rgba(161,0,255,.10) 16px 18px);
  mask: radial-gradient(180px 140px at -40px 20px, black 40%, transparent 41%) subtract,
        radial-gradient(180px 140px at calc(100% + 40px) calc(100% - 20px), black 40%, transparent 41%);
  pointer-events:none; mix-blend-mode:screen;
  animation: edges 6s linear infinite;
}
@keyframes edges{ to { background-position: 160px 0, 0 160px; } }
.btn{
  display:inline-block; padding: 12px 22px; font-weight:700; border-radius:14px;
  color:#051018; background: linear-gradient(90deg,var(--cyan),var(--sky),var(--mag));
  box-shadow: 0 0 18px rgba(0,255,225,.4), inset 0 0 12px rgba(255,255,255,.2);
  border:none; cursor:pointer; transition: transform .08s ease;
}
.btn:hover{ transform: translateY(-1px) scale(1.02); }
.small{font-size:.85rem; opacity:.8}
.bar{
  height:14px; border-radius:10px; background: rgba(255,255,255,.08);
  overflow:hidden; border:1px solid rgba(255,255,255,.15);
}
.fill{
  height:100%; background: linear-gradient(90deg,var(--cyan),var(--mag),var(--pink));
  box-shadow: 0 0 18px rgba(0,255,225,.35), 0 0 24px rgba(161,0,255,.25) inset;
  transition: width .5s ease;
}
.kpi{
  display:flex; align-items:center; gap:10px; margin:10px 0 8px;
}
.kpi span{ width:95px; opacity:.9 }
.overlay{
  position:fixed; inset:0; background: radial-gradient(70% 60% at 50% 50%, rgba(12,20,30,.8), rgba(5,10,18,.95));
  backdrop-filter: blur(6px);
  z-index:9999; display:flex; align-items:center; justify-content:center; flex-direction:column;
  color:#cfefff; text-align:center;
}
.glow{ text-shadow: 0 0 10px rgba(0,255,225,.6), 0 0 24px rgba(161,0,255,.45); }
.overlay small{ opacity:.75 }
.canvas-wrap{ width:min(920px, 90vw); height:min(520px, 60vh); border-radius:18px; overflow:hidden;
  border: 1px solid rgba(255,255,255,.15); box-shadow: 0 0 60px rgba(161,0,255,.25) inset; }
.note{ opacity:.8; font-size:.85rem }
</style>
""", unsafe_allow_html=True)

# ----------------------- HEADERS -----------------------
st.markdown('<div class="neon-title">Neuralink Hyperdrive — EEG Motor Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Demo • Not for clinical use</div>', unsafe_allow_html=True)
st.write("")

# ----------------------- INPUT GRID -----------------------
st.markdown('<div class="grid">', unsafe_allow_html=True)
st.markdown('<div class="card chip-edges left">', unsafe_allow_html=True)
st.markdown("### Enter EEG Feature Values")
st.markdown('<span class="note">Enter values in the ranges shown below. Tips: Fz=frontal, C3=left motor, Cz=central, C4=right motor.</span>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    fz_mean = st.slider("Fz_mean (−4 … +4)", -4.0, 4.0, 0.0, 0.1, help="Frontal mean amplitude")
    c3_mean = st.slider("C3_mean (−4 … +4)", -4.0, 4.0, 0.0, 0.1, help="Left motor cortex mean")
    c4_mean = st.slider("C4_mean (−4 … +4)", -4.0, 4.0, 0.0, 0.1, help="Right motor cortex mean")
with col2:
    fz_std  = st.slider("Fz_std (0.2 … 3.0)", 0.2, 3.0, 1.0, 0.1, help="Frontal variability (std)")
    cz_mean = st.slider("Cz_mean (−4 … +4)", -4.0, 4.0, 0.0, 0.1, help="Central mean")

st.markdown('<br/>', unsafe_allow_html=True)
go = st.button("Predict", type="primary", use_container_width=False)

st.markdown('</div>', unsafe_allow_html=True)  # close left card

# Right preview card
st.markdown('<div class="card chip-edges right">', unsafe_allow_html=True)
st.markdown("### Live Preview (Neon Brain)")
st.markdown(
    """
<div class="small">Idle preview — rotating neon brain (illustrative). The actual prediction uses your slider values only.</div>
""",
    unsafe_allow_html=True,
)

# Simple CSS animated “neon brain-ish” SVG preview
st.markdown(
    """
<div style="display:flex;justify-content:center;align-items:center; padding:6px 0 10px;">
  <svg width="420" height="260" viewBox="0 0 600 360">
    <defs>
      <linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#00FFE1"/><stop offset="50%" stop-color="#A100FF"/><stop offset="100%" stop-color="#FF007A"/>
      </linearGradient>
    </defs>
    <g transform="translate(300,180)">
      <g>
        <ellipse rx="180" ry="120" fill="none" stroke="url(#g1)" stroke-width="2"
                 style="filter: drop-shadow(0 0 8px rgba(0,255,225,.9));">
          <animateTransform attributeName="transform" attributeType="XML" type="rotate" from="0" to="360" dur="16s" repeatCount="indefinite"/>
        </ellipse>
        <ellipse rx="140" ry="90" fill="none" stroke="url(#g1)" stroke-width="2" opacity=".8">
          <animateTransform attributeName="transform" attributeType="XML" type="rotate" from="360" to="0" dur="13s" repeatCount="indefinite"/>
        </ellipse>
        <circle r="6" fill="#00FFE1">
          <animateMotion path="M 0 -110 C 120 -60, 120 60, 0 110 C -120 60, -120 -60, 0 -110 Z" dur="5s" repeatCount="indefinite"/>
        </circle>
        <circle r="5" fill="#A100FF">
          <animateMotion path="M 0 -80 C 90 -40, 90 40, 0 80 C -90 40, -90 -40, 0 -80 Z" dur="3.6s" repeatCount="indefinite"/>
        </circle>
      </g>
    </g>
  </svg>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)  # close right card
st.markdown('</div>', unsafe_allow_html=True)  # close grid

# ----------------------- PREDICTION LOGIC -----------------------
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)

def predict_probs(fz_mean, fz_std, c3_mean, cz_mean, c4_mean):
    # Heuristic demo “scores” (replace with real model later)
    score_left  = 1.2*c3_mean - 0.8*c4_mean + 0.2*cz_mean - 0.1*fz_mean
    score_right = 1.2*c4_mean - 0.8*c3_mean + 0.2*cz_mean - 0.1*fz_mean
    score_foot  = 1.5*cz_mean - 0.3*c3_mean - 0.3*c4_mean
    score_tong  = 1.0*fz_mean - 0.2*cz_mean + 0.05*(3.0 - abs(fz_std-1.0))  # tiny bonus if std ~1
    scores = np.array([score_left, score_right, score_foot, score_tong], dtype=float)
    probs = softmax(scores)
    labels = np.array(["Left","Right","Foot","Tongue"])
    pred_idx = int(np.argmax(probs))
    return labels, probs, labels[pred_idx]

def class_overlay_html(label:str) -> str:
    # Class-specific flourish in RESULT
    if label == "Left":
        direction = "translateX(-60px)"
        hue = "180deg"
    elif label == "Right":
        direction = "translateX(60px)"
        hue = "220deg"
    elif label == "Foot":
        direction = "translateY(60px)"
        hue = "140deg"
    else:  # Tongue
        direction = "translateY(-60px)"
        hue = "300deg"

    return f"""
<div style="position:relative; height:340px; border-radius:16px; overflow:hidden; border:1px solid rgba(255,255,255,.15);">
  <div style="position:absolute; inset:-30%; background:
    radial-gradient(60% 60% at 50% 50%, rgba(0,255,225,.25), transparent 60%),
    conic-gradient(from 0deg, rgba(161,0,255,.25), rgba(0,194,255,.25), rgba(255,0,122,.25), rgba(161,0,255,.25));
    filter:hue-rotate({hue}); animation: swirl 6s linear infinite;">
  </div>
  <div style="position:absolute; inset:0; display:flex; align-items:center; justify-content:center;">
    <div style="width:180px; height:180px; border-radius:50%;
         box-shadow: inset 0 0 60px rgba(0,255,225,.35), 0 0 40px rgba(161,0,255,.25);
         backdrop-filter: blur(8px); border:1px solid rgba(255,255,255,.15); transform:{direction};">
    </div>
  </div>
</div>
<style>
@keyframes swirl {{ to {{ transform: rotate(360deg); }} }}
</style>
"""

# ----------------------- RUN PREDICTION -----------------------
overlay = st.empty()
result_zone = st.container()

if go:
    # Full-screen loading overlay with a neon “hyperdrive”
    overlay.markdown("""
<div class="overlay">
  <div class="canvas-wrap">
    <!-- Lightweight CSS-only hyperdrive tunnel -->
    <div style="position:absolute; inset:0; background:
         radial-gradient(circle at 50% 50%, rgba(0,255,225,.25), transparent 40%),
         repeating-radial-gradient(circle at 50% 50%, rgba(161,0,255,.25) 0 6px, transparent 6px 12px);
         animation: zoom 2.2s ease-in forwards;"></div>
  </div>
  <h2 class="glow" style="margin-top:18px;">Loading…</h2>
  <small>Decoding signals</small>
</div>
<style>
@keyframes zoom{ 0%{transform:scale(1)} 80%{transform:scale(3)} 100%{transform:scale(5); opacity:.0} }
</style>
""", unsafe_allow_html=True)

    # Simulate compute latency
    time.sleep(2.2)
    overlay.empty()

    labels, probs, pred = predict_probs(fz_mean, fz_std, c3_mean, cz_mean, c4_mean)

    with result_zone:
        st.markdown(f"## Prediction: **:rainbow[{pred}]**")
        st.caption("Probabilities:")

        # Probability bars
        prob_df = pd.DataFrame({"Class":labels, "Prob":probs})
        for _, row in prob_df.iterrows():
            pct = int(round(row["Prob"]*100))
            st.markdown(
                f"""
<div class="kpi">
  <span>{row["Class"]}</span>
  <div class="bar" style="flex:1;">
    <div class="fill" style="width:{pct}%;"></div>
  </div>
  <span style="width:54px; text-align:right;">{pct}%</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("### Class-specific visual")
        st.markdown(class_overlay_html(pred), unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.button("Try Again", on_click=lambda: st.rerun())

# Helpful footer
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown(
    '<div class="small">Powered by demo heuristics. '
    'To plug in your real model later, replace the <code>predict_probs</code> function with model probabilities.</div>',
    unsafe_allow_html=True,
)




