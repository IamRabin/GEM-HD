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
pause_refresh_for_game = st.sidebar.checkbox("Pause auto-refresh during Mini Game", value=True)
st.sidebar.markdown("⏱ Auto-refresh enabled — watching `results/` folder")

# --------------------------------
# Path Configuration (project-relative)
# --------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
FIGURES_DIR = RESULTS_DIR / "figures"
LIVE_FRAME_FILE = FIGURES_DIR / "live.jpg"

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

def _file_info(path: Path):
    try:
        if path.exists():
            mtime = path.stat().st_mtime
            try:
                df = pd.read_parquet(path)
                n = len(df)
            except Exception:
                n = None
            return {"exists": True, "rows": n, "mtime": mtime}
    except Exception:
        pass
    return {"exists": False, "rows": None, "mtime": None}

def load_results():
    metrics = safe_load_json(METRICS_FILE)
    drift = safe_load_json(DRIFT_FILE)
    feedback = safe_load_json(FEEDBACK_FILE)
    return metrics, drift, feedback

def get_frame():
    """Capture a frame from webcam or fallback to placeholder."""
    # Prefer a live frame written by the streaming process to avoid webcam contention
    try:
        if LIVE_FRAME_FILE.exists():
            # If the live frame is too old (>5s), fall back to direct capture
            mtime = LIVE_FRAME_FILE.stat().st_mtime
            if time.time() - mtime < 5.0:
                return Image.open(LIVE_FRAME_FILE)
    except Exception:
        pass

    # Fallback: try reading directly from webcam
    try:
        cap = cv2.VideoCapture(0)
        success, frame = cap.read()
        cap.release()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame)
    except Exception:
        pass

    # Final fallback: placeholder image or gray canvas
    try:
        if PLACEHOLDER_IMAGE.exists():
            img = Image.open(PLACEHOLDER_IMAGE)
        else:
            # Fallback simple placeholder if image is missing
            img = Image.fromarray(np.ones((360, 640, 3), dtype=np.uint8) * 220)
    except Exception:
        try:
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
game_tab, tab1, tab2, tab3, tab4 = st.tabs(
    ["🎮 Mini Game", "🎥 Live Stream & CP Uncertainty", "📊 Drift Analysis", "💡 Feedback", "📜 Model Card"]
)

# --------------------------------
# Live Update Loop
# --------------------------------
while True:
    metrics, drift, feedback = load_results()

    # ---- Tab 0: Mini Game (Pinball-like) ----
    with game_tab:
        # Initialize state
        if "game_rounds_target" not in st.session_state:
            st.session_state["game_rounds_target"] = 5
        if "game_rounds_played" not in st.session_state:
            st.session_state["game_rounds_played"] = 0
        if "game_auto_launch" not in st.session_state:
            st.session_state["game_auto_launch"] = True
        if "game_focus" not in st.session_state:
            st.session_state["game_focus"] = False

        st.subheader("🎮 Pinball Mini Game")

        rounds_target = int(st.session_state["game_rounds_target"])  # for templating
        round_display = int(st.session_state["game_rounds_played"]) + 1
        auto_run = "true" if st.session_state["game_auto_launch"] else "false"

        # Always keep two columns; right is camera or placeholder
        col_game, col_cam = st.columns([2, 1])

        with col_game:
            st.components.v1.html(
                f"""
                <style>
                    #pinball-container {{
                        background: radial-gradient(circle at 50% 20%, #1f2937, #111827);
                        border-radius: 12px;
                        padding: 12px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
                    }}
                    #hud, #bottom-controls {{
                        color: #e5e7eb;
                        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
                    }}
                    #hud {{ display:flex; justify-content: space-between; align-items:center; margin-bottom: 8px; }}
                    #bottom-controls {{ display:flex; gap: 12px; align-items: center; margin-top: 10px; flex-wrap: wrap; }}
                    .control {{ background:#2563eb; color:#fff; border:none; padding:8px 12px; border-radius:8px; cursor:pointer; }}
                    .control:active {{ transform: translateY(1px); }}
                    #help {{ color: #9ca3af; font-size: 13px; margin-top: 4px; }}
                    #power {{ height: 10px; background:#111827; border-radius:6px; overflow:hidden; border:1px solid #1f2937; }}
                    #power > div {{ height:100%; width:0%; background:linear-gradient(90deg, #22c55e, #f59e0b, #ef4444); transition: width 0.05s; }}
                </style>
                <div id="pinball-container">
                    <div id=\"hud\"><div>Score: <span id=\"score\">0</span></div><div>Round: <span id=\"round\">{round_display}</span> / {rounds_target}</div></div>
                    <canvas id="board" width="480" height="720" tabindex="0"></canvas>
                    <div id="help">Controls: Left/Right arrows for flippers · Space to launch (hold for power) · P pause · R reset.</div>
                    <div id="bottom-controls">
                        <div style=\"min-width:200px;\">
                            <div style=\"color:#9ca3af; font-size:12px; margin-bottom:4px;\">Launch Power</div>
                            <div id=\"power\"><div id=\"bar\"></div></div>
                        </div>
                        <button class=\"control\" onclick=\"launch()\">Space/Launch</button>
                        <button class=\"control\" onclick=\"resetGame()\">R/Reset</button>
                        <button class=\"control\" onclick=\"togglePause()\">P/Pause</button>
                    </div>
                </div>
                <script>
                (function() {{
                    const autoRun = {auto_run};
                    const canvas = document.getElementById('board');
                    const ctx = canvas.getContext('2d');
                    const W = canvas.width, H = canvas.height;
                    let score = 0;
                    let paused = false;
                    let round = {round_display};
                    const gravity = 0.25;
                    const damping = 0.82;
                    const ball = {{ x: W-40, y: H-40, vx: 0, vy: 0, r: 9, inPlay: false }};
                    const keys = {{ left: false, right: false }};
                    // Power charge
                    let charging = false; let power = 0; const maxPower = 1.0; const bar = document.getElementById('bar');

                    // Flippers
                    const flipperWidth = 12, flipperLen = 70;
                    const leftFlipper = {{
                        px: 100, py: H - 120, angle: Math.PI * 0.22, rest: Math.PI * 0.22, active: Math.PI * -0.10, vel: 0
                    }};
                    const rightFlipper = {{
                        px: W - 100, py: H - 120, angle: Math.PI * (1 - 0.22), rest: Math.PI * (1 - 0.22), active: Math.PI * (1 + 0.10), vel: 0
                    }};

                    // Bumpers
                    const bumpers = [
                        {{x: 120, y: 160, r: 16, s: 50}},
                        {{x: 240, y: 120, r: 18, s: 75}},
                        {{x: 360, y: 180, r: 14, s: 40}},
                        {{x: 160, y: 260, r: 20, s: 60}},
                        {{x: 300, y: 310, r: 16, s: 50}},
                        {{x: 240, y: 420, r: 22, s: 90}}
                    ];
                    const leftWallX = 30, rightWallX = W - 30, laneX = W - 70;

                    // Space-cadet style theming helpers
                    const stars = Array.from({{length: 80}}, () => ({{ x: Math.random()*W, y: Math.random()*H, r: Math.random()*1.5+0.3 }}));
                    function drawTheme() {{
                        // Starfield
                        for (const s of stars) {{
                            ctx.fillStyle = 'rgba(255,255,255,0.85)';
                            ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, Math.PI*2); ctx.fill();
                        }}
                        // Bottom plasma glow
                        const grad = ctx.createRadialGradient(W*0.5, H*0.98, 10, W*0.5, H*0.8, 220);
                        grad.addColorStop(0, 'rgba(168,85,247,0.35)');
                        grad.addColorStop(1, 'rgba(168,85,247,0)');
                        ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(W*0.5, H*0.98, 240, Math.PI, Math.PI*2); ctx.fill();
                        // Neon rails
                        ctx.strokeStyle = '#8b5cf6'; ctx.lineWidth = 4; ctx.beginPath();
                        ctx.moveTo(leftWallX+8, H-220); ctx.quadraticCurveTo(80, H-420, 70, 120); ctx.stroke();
                        ctx.beginPath(); ctx.moveTo(rightWallX-8, H-220); ctx.quadraticCurveTo(W-80, H-420, W-70, 120); ctx.stroke();
                        // Inlane arrows
                        ctx.fillStyle = '#10b981';
                        ctx.beginPath(); ctx.moveTo(70, H-150); ctx.lineTo(110, H-120); ctx.lineTo(70, H-110); ctx.closePath(); ctx.fill();
                        ctx.beginPath(); ctx.moveTo(W-70, H-150); ctx.lineTo(W-110, H-120); ctx.lineTo(W-70, H-110); ctx.closePath(); ctx.fill();
                        // Center ring lights
                        const cx = W*0.5, cy = H*0.55, R = 70;
                        for (let i=0;i<16;i++) {{
                            const a = i/16 * Math.PI*2; const px = cx + Math.cos(a)*R; const py = cy + Math.sin(a)*R;
                            ctx.fillStyle = i%2? '#fde047' : '#fb7185'; ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI*2); ctx.fill();
                        }}
                    }}

                    function drawBumper(b) {{
                        // Bullseye bumper with glow
                        const g = ctx.createRadialGradient(b.x, b.y, 2, b.x, b.y, b.r*1.4);
                        g.addColorStop(0, 'rgba(59,130,246,0.6)');
                        g.addColorStop(1, 'rgba(59,130,246,0)');
                        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(b.x, b.y, b.r*1.4, 0, Math.PI*2); ctx.fill();
                        ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.arc(b.x, b.y, b.r, 0, Math.PI*2); ctx.fill();
                        ctx.fillStyle = '#2563eb'; ctx.beginPath(); ctx.arc(b.x, b.y, b.r*0.7, 0, Math.PI*2); ctx.fill();
                        ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.arc(b.x, b.y, b.r*0.42, 0, Math.PI*2); ctx.fill();
                        ctx.fillStyle = '#ef4444'; ctx.beginPath(); ctx.arc(b.x, b.y, b.r*0.22, 0, Math.PI*2); ctx.fill();
                    }}

                    function launch() {{
                        if (!ball.inPlay) {{
                            const p = Math.min(1, Math.max(0.2, power || (autoRun ? 0.6 : 0.4)));
                            const v = 8 + 10*p; // 8..18
                            ball.x = W - 40; ball.y = H - 40; ball.vx = -3.2 - 2.0*p; ball.vy = -v; ball.inPlay = true; power = 0; bar.style.width = '0%';
                        }}
                    }}
                    window.launch = launch;

                    window.resetGame = function() {{
                        score = 0; document.getElementById('score').textContent = score;
                        round = 1; document.getElementById('round').textContent = round;
                        ball.inPlay = false; ball.vx = ball.vy = 0; ball.x = W - 40; ball.y = H - 40;
                    }}

                    window.togglePause = function() {{ paused = !paused; }}

                    function reflect(vx, vy, nx, ny) {{
                        const dot = vx*nx + vy*ny;
                        return [vx - 2*dot*nx, vy - 2*dot*ny];
                    }}

                    function segNearestPoint(px, py, qx, qy, x, y) {{
                        const dx = qx - px, dy = qy - py;
                        const t = Math.max(0, Math.min(1, ((x - px)*dx + (y - py)*dy) / ((dx*dx + dy*dy)||1)));
                        return [px + t*dx, py + t*dy, t];
                    }}

                    function collideFlipper(flip, kickDir) {{
                        const tx = flip.px + Math.cos(flip.angle) * flipperLen;
                        const ty = flip.py + Math.sin(flip.angle) * flipperLen;
                        const [nx, ny, t] = segNearestPoint(flip.px, flip.py, tx, ty, ball.x, ball.y);
                        const dx = ball.x - nx, dy = ball.y - ny;
                        const dist = Math.hypot(dx, dy);
                        if (dist < ball.r + flipperWidth*0.5 && t > 0.05 && t < 0.95) {{
                            const nux = dx / (dist||1), nuy = dy / (dist||1);
                            [ball.vx, ball.vy] = reflect(ball.vx, ball.vy, nux, nuy);
                            const kick = 4.5;
                            ball.vx += kick * (-Math.sin(flip.angle)) * kickDir;
                            ball.vy += kick * ( Math.cos(flip.angle)) * kickDir;
                            ball.x = nx + (ball.r + flipperWidth*0.5 + 0.5) * nux;
                            ball.y = ny + (ball.r + flipperWidth*0.5 + 0.5) * nuy;
                            score += 5; document.getElementById('score').textContent = score;
                        }}
                    }}

                    function drawFlipper(flip) {{
                        const tx = flip.px + Math.cos(flip.angle) * flipperLen;
                        const ty = flip.py + Math.sin(flip.angle) * flipperLen;
                        ctx.strokeStyle = '#ff3b30';
                        ctx.lineWidth = flipperWidth;
                        ctx.lineCap = 'round';
                        ctx.beginPath();
                        ctx.moveTo(flip.px, flip.py);
                        ctx.lineTo(tx, ty);
                        ctx.stroke();
                    }}

                    function step() {{
                        if (paused) {{ requestAnimationFrame(step); return; }}
                        ctx.clearRect(0,0,W,H);
                        ctx.fillStyle = '#0b1220'; ctx.fillRect(0,0,W,H);
                        drawTheme();
                        ctx.fillStyle = '#374151';
                        ctx.fillRect(leftWallX-6, 0, 12, H);
                        ctx.fillRect(rightWallX-6, 0, 12, H);
                        ctx.fillStyle = '#4b5563'; ctx.fillRect(laneX-2, H-220, 4, 220);

                        for (const b of bumpers) {{ drawBumper(b); }}

                        const leftTarget = keys.left ? leftFlipper.active : leftFlipper.rest;
                        const rightTarget = keys.right ? rightFlipper.active : rightFlipper.rest;
                        leftFlipper.angle += (leftTarget - leftFlipper.angle) * 0.25;
                        rightFlipper.angle += (rightTarget - rightFlipper.angle) * 0.25;

                        if (ball.inPlay) {{
                            ball.vy += gravity;
                            ball.x += ball.vx; ball.y += ball.vy;
                            if (ball.x - ball.r < leftWallX) {{ ball.x = leftWallX + ball.r; ball.vx = -ball.vx * damping; }}
                            if (ball.x + ball.r > rightWallX) {{ ball.x = rightWallX - ball.r; ball.vx = -ball.vx * damping; }}
                            if (ball.y - ball.r < 10) {{ ball.y = 10 + ball.r; ball.vy = -ball.vy * damping; }}

                            for (const b of bumpers) {{
                                const dx = ball.x - b.x, dy = ball.y - b.y; const dist = Math.hypot(dx, dy);
                                if (dist < ball.r + b.r) {{
                                    const nx = dx / (dist||1), ny = dy / (dist||1);
                                    [ball.vx, ball.vy] = reflect(ball.vx, ball.vy, nx, ny);
                                    ball.vx *= 0.98; ball.vy *= 0.98; score += b.s; document.getElementById('score').textContent = score;
                                    ball.x = b.x + (ball.r + b.r + 0.5) * nx; ball.y = b.y + (ball.r + b.r + 0.5) * ny;
                                }}
                            }}

                            collideFlipper(leftFlipper, +1);
                            collideFlipper(rightFlipper, -1);

                            if (ball.y - ball.r > H) {{
                                ball.inPlay = false; ball.vx = ball.vy = 0; ball.x = W - 40; ball.y = H - 40;
                                round += 1; document.getElementById('round').textContent = round;
                                if (autoRun) {{ setTimeout(() => {{ power = 0.6; bar.style.width = '60%'; launch(); }}, 800); }}
                            }}
                        }}

                        drawFlipper(leftFlipper); drawFlipper(rightFlipper);
                        ctx.beginPath(); const gb = ctx.createRadialGradient(ball.x-3, ball.y-3, 2, ball.x, ball.y, ball.r);
                        gb.addColorStop(0, '#f3f4f6'); gb.addColorStop(1, '#9ca3af'); ctx.fillStyle = gb; ctx.arc(ball.x, ball.y, ball.r, 0, Math.PI*2); ctx.fill();

                        requestAnimationFrame(step);
                    }}

                    function kd(e) {{
                        if (e.code === 'ArrowLeft' || e.code === 'KeyA') keys.left = true;
                        if (e.code === 'ArrowRight' || e.code === 'KeyD') keys.right = true;
                        if (e.code === 'Space') {{ if (!charging && !ball.inPlay) {{ charging = true; }} }}
                        if (e.code === 'KeyP') paused = !paused;
                        if (e.code === 'KeyR') window.resetGame();
                    }}
                    function ku(e) {{
                        if (e.code === 'ArrowLeft' || e.code === 'KeyA') keys.left = false;
                        if (e.code === 'ArrowRight' || e.code === 'KeyD') keys.right = false;
                        if (e.code === 'Space') {{ if (charging) {{ charging = false; launch(); }} }}
                    }}
                    window.addEventListener('keydown', kd);
                    window.addEventListener('keyup', ku);
                    canvas.addEventListener('click', () => canvas.focus());

                    function powerLoop() {{
                        if (charging) {{ power = Math.min(maxPower, power + 0.02); }} else {{ power = Math.max(0, power - 0.03); }}
                        bar.style.width = Math.round(100*power) + '%';
                        requestAnimationFrame(powerLoop);
                    }}

                    powerLoop();
                    if (autoRun) {{ setTimeout(() => {{ power = 0.6; bar.style.width = '60%'; launch(); }}, 500); }}
                    step();
                }})();
                </script>
                """,
                height=800,
                scrolling=False,
            )

        with col_cam:
            st.subheader("🎥 Live Camera During Game")
            # Fixed layout: show camera or blank placeholder (no fade) to keep position
            if st.session_state.get("game_focus", False):
                st.image(np.zeros((360, 640, 3), dtype=np.uint8) + 16, caption="Live feed hidden", use_container_width=True)
            else:
                frame = get_frame()
                st.image(frame, caption="Live gaze feed", use_container_width=True)
            st.caption("Follow the ball with your eyes while the camera records as usual.")

        # Bottom control bar (moved to end per spec)
        st.markdown("---")
        bcols = st.columns([2, 1, 1, 1, 2])
        with bcols[0]:
            st.session_state["game_rounds_target"] = st.number_input(
                "Rounds to play", min_value=1, max_value=20, value=int(st.session_state["game_rounds_target"]), step=1
            )
        with bcols[1]:
            if st.button("Next Round"):
                st.session_state["game_rounds_played"] = min(
                    st.session_state["game_rounds_played"] + 1,
                    st.session_state["game_rounds_target"],
                )
        with bcols[2]:
            if st.button("Reset Rounds"):
                st.session_state["game_rounds_played"] = 0
        with bcols[3]:
            st.metric("Rounds", f"{st.session_state['game_rounds_played']} / {st.session_state['game_rounds_target']}")
        with bcols[4]:
            c1, c2 = st.columns(2)
            with c1:
                st.session_state["game_auto_launch"] = st.toggle("Auto-launch", value=st.session_state["game_auto_launch"], key="game_auto_launch_ctl")
            with c2:
                st.session_state["game_focus"] = st.toggle("Hide camera", value=st.session_state["game_focus"], key="game_focus_ctl")

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

            # Stream health diagnostics
            st.markdown("---")
            st.caption("Stream health (diagnostics)")
            cur_path = BASE_DIR / "data/processed/current.parquet"
            pred_path = find_current_with_pred() or (BASE_DIR / "data/processed/current_with_pred.parquet")
            cur_info = _file_info(cur_path)
            pred_info = _file_info(pred_path) if pred_path is not None else {"exists": False, "rows": None, "mtime": None}
            frame_age = None
            try:
                if LIVE_FRAME_FILE.exists():
                    frame_age = time.time() - LIVE_FRAME_FILE.stat().st_mtime
            except Exception:
                pass
            c1, c2, c3 = st.columns(3)
            c1.metric("current.parquet rows", cur_info["rows"] if cur_info["rows"] is not None else 0)
            c2.metric("current_with_pred rows", pred_info["rows"] if pred_info["rows"] is not None else 0)
            c3.metric("Live frame age (s)", f"{frame_age:.1f}" if frame_age is not None else "n/a")

            if use_live_preds and (not pred_info["exists"] or (pred_info["rows"] is not None and pred_info["rows"] <= 0)):
                st.warning("No live predictions detected. Start streaming: 1) webcam features, 2) streaming inference.")
                st.code(
                    """
python -m src.webcam_stream_features --output data/processed/current.parquet
python -m src.stream_infer --ref data/processed/ref.parquet --current data/processed/current.parquet --output data/processed/current_with_pred.parquet --interval 1.0
""".strip(),
                    language="bash",
                )

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

    # Extend refresh interval while game is running, if requested
    _effective_refresh = refresh_interval
    if pause_refresh_for_game and st.session_state.get("game_running", False):
        _effective_refresh = max(60, refresh_interval * 20)
    time.sleep(_effective_refresh)
    st.rerun()
