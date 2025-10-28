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
use_live_preds = st.sidebar.checkbox("Use live predictions (from data/processed)", value=True)
st.sidebar.markdown("⏱ Auto-refresh enabled — watching `results/` folder")

# --------------------------------
# Path Configuration (project-relative)
# --------------------------------
BASE_DIR = Path(__file__).resolve().parent
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

def find_current_with_pred():
    candidates = [
        BASE_DIR / "data/processed/current_with_pred.parquet",
        BASE_DIR / "src/evaluation_agent/data/processed/current_with_pred.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def read_latest_prediction():
    p = find_current_with_pred()
    if p is None:
        return None
    try:
        df = pd.read_parquet(p)
        if len(df) == 0:
            return None
        row = df.iloc[-1]
        y_pred = float(row.get("y_pred", np.nan))
        y_std = row.get("y_std", np.nan)
        y_std = float(y_std) if pd.notnull(y_std) else None
        if np.isnan(y_pred):
            return None
        return {"y_pred": y_pred, "y_std": y_std}
    except Exception:
        return None

def read_last_n_predictions(n: int):
    p = find_current_with_pred()
    if p is None:
        return None
    try:
        df = pd.read_parquet(p)
        if len(df) == 0:
            return None
        tail = df.tail(n)
        # Keep only valid numeric rows
        tail = tail[pd.to_numeric(tail.get("y_pred", np.nan), errors="coerce").notna()]
        if len(tail) == 0:
            return None
        y_pred = tail["y_pred"].astype(float).tolist()
        y_std = tail.get("y_std")
        y_std = y_std.astype(float).tolist() if y_std is not None else None
        return {"y_pred": y_pred, "y_std": y_std}
    except Exception:
        return None

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
        try:
            if PLACEHOLDER_IMAGE.exists():
                img = Image.open(PLACEHOLDER_IMAGE)
            else:
                # Fallback simple placeholder if image is missing
                img = Image.fromarray(np.ones((360, 640, 3), dtype=np.uint8) * 220)
        except Exception:
            img = Image.fromarray(np.ones((360, 640, 3), dtype=np.uint8) * 220)
    return img

"""
Session state initialization to persist model and buffers across reruns
"""
# Persist conformal regressor across reruns
if "conf_reg" not in st.session_state:
    st.session_state["conf_reg"] = ConformalRegressor(alpha=alpha)
    st.session_state["conf_alpha"] = alpha

# Update alpha if changed
if st.session_state.get("conf_alpha") != alpha:
    st.session_state["conf_alpha"] = alpha
    st.session_state["conf_reg"].alpha = alpha

# Calibrate once (or re-use if previously calibrated)
if not getattr(st.session_state["conf_reg"], "calibrated", False):
    calib_paths = [
        BASE_DIR / "data/processed/current_with_pred.parquet",
        BASE_DIR / "data/processed/ref.parquet",
        BASE_DIR / "src/evaluation_agent/data/processed/ref.parquet",
    ]
    calibrated = False
    for p in calib_paths:
        try:
            if p.exists():
                df = pd.read_parquet(p)
                if {"ema_engagement", "y_pred"}.issubset(df.columns):
                    st.session_state["conf_reg"].fit_calibration(df["ema_engagement"], df["y_pred"])
                    calibrated = True
                    break
        except Exception:
            pass
    if not calibrated:
        st.session_state["conf_reg"].fit_calibration(np.random.rand(200), np.random.rand(200))

# Persist rolling buffers across reruns and handle window size changes
def _ensure_buffers(maxlen: int):
    if "buffer_maxlen" not in st.session_state:
        st.session_state["buffer_maxlen"] = maxlen
    if (
        "y_pred_buffer" not in st.session_state
        or "y_lower_buffer" not in st.session_state
        or "y_upper_buffer" not in st.session_state
    ):
        st.session_state["y_pred_buffer"] = deque(maxlen=maxlen)
        st.session_state["y_lower_buffer"] = deque(maxlen=maxlen)
        st.session_state["y_upper_buffer"] = deque(maxlen=maxlen)
        st.session_state["buffer_maxlen"] = maxlen
    elif st.session_state["buffer_maxlen"] != maxlen:
        # Resize while keeping recent history
        def _resize(dq: deque, new_maxlen: int) -> deque:
            tmp = list(dq)[-new_maxlen:]
            ndq = deque(maxlen=new_maxlen)
            ndq.extend(tmp)
            return ndq
        st.session_state["y_pred_buffer"] = _resize(st.session_state["y_pred_buffer"], maxlen)
        st.session_state["y_lower_buffer"] = _resize(st.session_state["y_lower_buffer"], maxlen)
        st.session_state["y_upper_buffer"] = _resize(st.session_state["y_upper_buffer"], maxlen)
        st.session_state["buffer_maxlen"] = maxlen

_ensure_buffers(window_size)

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
            st.subheader("📈 Live Engagement Prediction + Uncertainty")

            # Get recent predictions: use last N rows if live, otherwise simulate one
            if use_live_preds:
                recent = read_last_n_predictions(window_size)
            else:
                recent = None

            if recent is not None:
                preds = recent["y_pred"]
                stds = recent.get("y_std")
                # Rebuild buffers from recent window
                st.session_state["y_pred_buffer"].clear(); st.session_state["y_lower_buffer"].clear(); st.session_state["y_upper_buffer"].clear()
                for i, pval in enumerate(preds):
                    if stds is not None and i < len(stds) and stds[i] is not None and not np.isnan(stds[i]):
                        q = float(stds[i])
                    else:
                        q = st.session_state["conf_reg"].get_quantile()
                    y_lower, y_upper = pval - q, pval + q
                    st.session_state["y_pred_buffer"].append(float(pval))
                    st.session_state["y_lower_buffer"].append(max(0, y_lower))
                    st.session_state["y_upper_buffer"].append(min(1, y_upper))
            else:
                engagement_pred = np.random.uniform(0.4, 0.9)
                q = st.session_state["conf_reg"].get_quantile()
                y_lower, y_upper = engagement_pred - q, engagement_pred + q
                st.session_state["y_pred_buffer"].append(engagement_pred)
                st.session_state["y_lower_buffer"].append(max(0, y_lower))
                st.session_state["y_upper_buffer"].append(min(1, y_upper))

            # Plot
            fig, ax = plt.subplots()
            x = np.arange(len(st.session_state["y_pred_buffer"]))
            y_pred_series = list(st.session_state["y_pred_buffer"]) or [np.nan]
            y_low_series = list(st.session_state["y_lower_buffer"]) or [np.nan]
            y_up_series = list(st.session_state["y_upper_buffer"]) or [np.nan]

            ax.plot(x, y_pred_series, color="royalblue", linewidth=2.0, label="Predicted Engagement (0–1)")
            ax.fill_between(x, y_low_series, y_up_series, color="lightblue", alpha=0.35, label="Prediction Interval (±q)")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Engagement score")
            ax.set_xlabel("Time steps")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper left")
            st.pyplot(fig)

            # Metrics and risk badge
            if len(y_pred_series) > 0 and not np.isnan(y_pred_series[-1]):
                last_pred = float(y_pred_series[-1])
                last_width = float(max(0.0, min(1.0, y_up_series[-1] - y_low_series[-1]))) if not np.isnan(y_up_series[-1]) else None
                cols = st.columns(3)
                cols[0].metric("Latest engagement", f"{last_pred:.3f}")
                if last_width is not None:
                    cols[1].metric("Interval width", f"{last_width:.3f}")
                # Simple risk label
                def risk_label(y, w):
                    if y < 0.3 and (w is None or w < 0.25):
                        return "HIGH"
                    if y < 0.5 and (w is not None and w >= 0.25):
                        return "MODERATE"
                    if y >= 0.7 and (w is None or w < 0.15):
                        return "LOW"
                    return "UNCERTAIN"
                r = risk_label(last_pred, last_width)
                cols[2].markdown(f"**Risk level:** `{r}`")

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
    # Clarify what is being predicted
    st.caption("Predicted engagement is a normalized score in [0,1] estimating user engagement from gaze features. The shaded band shows an uncertainty interval (conformal half-width q or provided y_std).")

    time.sleep(refresh_interval)
    st.rerun()
