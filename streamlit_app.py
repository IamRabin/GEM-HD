import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, time, cv2
from pathlib import Path
from PIL import Image
from collections import deque

from src.conformal.conformal_utils import ConformalRegressor


# --------------------------------
# Streamlit Configuration
# --------------------------------
st.set_page_config(page_title="GEM-HD Live Dashboard", layout="wide")
st.title("👁️ GEM-HD: Gaze-based Early Mental Health Detection")

# --------------------------------
# Sidebar Controls
# --------------------------------
st.sidebar.header("Controls")
refresh_interval = st.sidebar.slider("Refresh interval (sec)", 2, 15, 5)
alpha = st.sidebar.slider("Conformal α (uncertainty level)", 0.01, 0.3, 0.1)
window_size = st.sidebar.slider("Rolling window size", 20, 100, 50)
st.sidebar.markdown("⏱ Auto-refresh enabled — watching `results/` folder")

# --------------------------------
# Path Configuration
# --------------------------------
BASE_DIR = Path("/home/rabink1/D1/gemhd/GEM-HD")
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
FIGURES_DIR = RESULTS_DIR / "figures"

METRICS_FILE = RESULTS_DIR / "metrics.json"
DRIFT_FILE = RESULTS_DIR / "drift_report.json"
FEEDBACK_FILE = RESULTS_DIR / "feedback_report.json"
MODEL_CARD_FILE = RESULTS_DIR / "model_card_V.1.0.html"
PLACEHOLDER_IMAGE = FIGURES_DIR / "GEM.png"

# --------------------------------
# Helper functions
# --------------------------------
def safe_load_json(path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as e:
            st.warning(f"⚠️ Error reading {path.name}: {e}")
            return {}
    return {}

def load_results():
    metrics = safe_load_json(METRICS_FILE)
    drift = safe_load_json(DRIFT_FILE)
    feedback = safe_load_json(FEEDBACK_FILE)
    return metrics, drift, feedback

def get_frame():
    """Capture a frame from webcam or fallback to placeholder."""
    cap = cv2.VideoCapture(0)
    success, frame = cap.read()
    cap.release()
    if success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
    else:
        img = Image.open(PLACEHOLDER_IMAGE)
    return img

# --------------------------------
# Initialize Conformal Model and Buffers
# --------------------------------
conf_reg = ConformalRegressor(alpha=alpha)
# Use dummy calibration if needed
try:
    ref_df = pd.read_parquet(BASE_DIR / "src/evaluation_agent/data/processed/ref.parquet")
    conf_reg.fit_calibration(ref_df["ema_engagement"], ref_df["y_pred"])
except Exception:
    conf_reg.fit_calibration(np.random.rand(200), np.random.rand(200))

y_pred_buffer = deque(maxlen=window_size)
y_lower_buffer = deque(maxlen=window_size)
y_upper_buffer = deque(maxlen=window_size)

# --------------------------------
# Tabs Layout
# --------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎥 Live Stream & CP Uncertainty", "📊 Drift Analysis", "💡 Feedback", "📜 Model Card"]
)

# --------------------------------
# Live Update Loop
# --------------------------------
while True:
    metrics, drift, feedback = load_results()

    # ---- Tab 1: Webcam + Live Conformal Plot ----
    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🎥 Live Gaze Stream")
            frame = get_frame()
            st.image(frame, caption="Live or simulated gaze feed", use_container_width=True)

        with col2:
            st.subheader("📈 Live Conformal Uncertainty Plot")

            # Simulate a new engagement prediction
            engagement_pred = np.random.uniform(0.4, 0.9)
            q = conf_reg.get_quantile()
            y_lower, y_upper = engagement_pred - q, engagement_pred + q

            y_pred_buffer.append(engagement_pred)
            y_lower_buffer.append(max(0, y_lower))
            y_upper_buffer.append(min(1, y_upper))

            # Plot
            fig, ax = plt.subplots()
            x = np.arange(len(y_pred_buffer))
            ax.plot(x, list(y_pred_buffer), color="royalblue", label="Predicted Engagement")
            ax.fill_between(x, list(y_lower_buffer), list(y_upper_buffer),
                            color="lightblue", alpha=0.4, label="CP Interval")
            ax.set_ylim(0, 1)
            ax.set_xlabel("Time Steps")
            ax.set_title("Live Conformal Prediction Uncertainty")
            ax.legend()
            st.pyplot(fig)

    # ---- Tab 2: Drift ----
    with tab2:
        st.subheader("🧩 Feature Drift Monitoring")
        if drift:
            ks = drift.get("ks_report", {})
            if ks:
                drift_df = pd.DataFrame([
                    {"Feature": f, "Drift Detected": d["drift_detected"], "p_value": d["p_value"]}
                    for f, d in ks.items()
                ])
                st.dataframe(drift_df, use_container_width=True)
                bar_data = drift_df.set_index("Feature")["p_value"]
                st.bar_chart(1 - bar_data)
            st.metric("Outlier Ratio", f"{drift.get('outlier_ratio', 0):.3f}")
        else:
            st.info("Waiting for drift_report.json...")

    # ---- Tab 3: Feedback ----
    with tab3:
        st.subheader("💡 Feedback Agent Recommendations")
        if feedback:
            actions = feedback.get("actions", [])
            if actions:
                st.table(pd.DataFrame(actions))
            else:
                st.success("✅ Model stable — no corrective actions required.")
        else:
            st.info("Waiting for feedback_report.json...")

    # ---- Tab 4: Model Card ----
    with tab4:
        st.subheader("📜 Model Card (Latest Version)")
        if MODEL_CARD_FILE.exists():
            st.components.v1.html(MODEL_CARD_FILE.read_text(), height=600, scrolling=True)
        else:
            st.info("Run the full pipeline to generate model card HTML.")

    # ---- Refresh every few seconds ----
    time.sleep(refresh_interval)
    st.rerun()
