"""
Real-time Emotion Detector + Emotion History Log
Run:  streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime

from src.emotion_detector import detect_emotion, blur_background

# Page config
st.set_page_config(
    page_title="MoodLens",
    page_icon=":material/face:",
    layout="wide",
)

# Session state: emotion history
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"time", "emotion", "confidence"}

# Theme
st.markdown("""
<style>
.stApp { background-color: #0e0e0e; }
[data-testid="stSidebar"] { background-color: rgba(26,26,26,0.85); border-right: 1px solid #2a2a2a; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); }
html, body, [class*="css"] { color: #e0e0e0; }
h1, h2, h3 { color: #ffffff !important; }
[data-testid="stMetric"] { background-color: rgba(30,30,30,0.8); border: 1px solid #2e2e2e; border-radius: 8px; padding: 16px; backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); }
[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: #888 !important; }
.stButton > button { background-color: #2a2a2a; color: #e0e0e0; border: 1px solid #3a3a3a; border-radius: 6px; }
.stButton > button:hover { background-color: #3a3a3a; border-color: #555; }
.stSlider [data-baseweb="slider"] { background-color: #2a2a2a; }
.stAlert { background-color: #1e1e1e !important; border-color: #333 !important; }
.stRadio label { color: #ccc !important; }
[data-testid="stFileUploader"] { background-color: #1a1a1a; border: 1px dashed #3a3a3a; border-radius: 8px; }
hr { border-color: #2a2a2a; }
.stCaption { color: #666 !important; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:4px">
  <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24"
       fill="none" stroke="#aaa" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="10"/>
    <path d="M8 14s1.5 2 4 2 4-2 4-2"/>
    <line x1="9" y1="9" x2="9.01" y2="9"/>
    <line x1="15" y1="9" x2="15.01" y2="9"/>
  </svg>
  <span style="font-size:2rem; font-weight:700; color:#fff; letter-spacing:-0.5px">MoodLens</span>
</div>
<p style="color:#666; margin-top:0; margin-bottom:24px; font-size:0.9rem">
  Real-time emotion detection with mood history
</p>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px">
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
           fill="none" stroke="#aaa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="3"/>
        <path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/>
      </svg>
      <span style="font-weight:600; font-size:1rem; color:#ddd">Settings</span>
    </div>
    """, unsafe_allow_html=True)
    confidence_threshold = st.slider("Confidence threshold", 0.1, 1.0, 0.4, 0.05)
    st.markdown("---")
    blur_bg = st.toggle("Blur background", value=True)
    blur_strength = st.slider("Blur strength", 5, 61, 31, 2) if blur_bg else 31
    st.markdown("---")
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

# Layout
col_cam, col_result = st.columns([3, 2])

with col_cam:
    source = st.radio("Input source", ["Webcam", "Upload photo"], horizontal=True)
    frame_placeholder  = st.empty()
    status_placeholder = st.empty()

with col_result:
    emotion_placeholder = st.empty()
    history_placeholder = st.empty()

# Constants
EMOTION_COLORS = {
    "happy": "#4ade80", "sad": "#60a5fa", "angry": "#f87171",
    "fear": "#c084fc",  "surprise": "#fb923c", "disgust": "#a3e635",
    "neutral": "#94a3b8",
}

def emotion_icon_html(emotion: str, size: int = 48) -> str:
    color = EMOTION_COLORS.get(emotion, "#aaa")
    mouth = (
        '<path d="M8 14s1.5 2 4 2 4-2 4-2"/>' if emotion == "happy" else
        '<path d="M16 16s-1.5-2-4-2-4 2-4 2"/>' if emotion in ("sad", "angry", "disgust") else
        '<path d="M8 15h8"/>'
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24"
       fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="10"/>{mouth}
      <line x1="9" y1="9" x2="9.01" y2="9" stroke-width="2.5"/>
      <line x1="15" y1="9" x2="15.01" y2="9" stroke-width="2.5"/>
    </svg>"""

def render_history():
    if not st.session_state.history:
        history_placeholder.markdown(
            '<div style="color:#444;font-size:0.85rem;margin-top:12px">No detections yet.</div>',
            unsafe_allow_html=True,
        )
        return
    rows = []
    for entry in reversed(st.session_state.history[-10:]):
        accent = EMOTION_COLORS.get(entry["emotion"], "#aaa")
        dot  = f'<span style="width:9px;height:9px;border-radius:50%;background:{accent};display:inline-block;flex-shrink:0"></span>'
        name = f'<span style="flex:1;color:{accent};font-weight:600;font-size:0.88rem">{entry["emotion"].capitalize()}</span>'
        conf = f'<span style="color:#555;font-size:0.78rem">{entry["confidence"]:.0%}</span>'
        ts   = f'<span style="color:#444;font-size:0.75rem;margin-left:10px">{entry["time"]}</span>'
        rows.append(
            '<div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #1e1e1e">'
            + dot + name + conf + ts + '</div>'
        )
    html = (
        '<div style="margin-top:12px">'
        '<div style="color:#666;font-size:0.78rem;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.05em">Recent detections</div>'
        + "".join(rows) + '</div>'
    )
    history_placeholder.markdown(html, unsafe_allow_html=True)
def process_frame(img_rgb: np.ndarray):
    result = detect_emotion(img_rgb, confidence_threshold)

    if result is None:
        status_placeholder.warning("No face detected — make sure your face is visible.")
        emotion_placeholder.empty()
        render_history()
        return

    x, y, w, h = result["box"]
    emotion    = result["emotion"]
    confidence = result["confidence"]

    # Annotate frame
    annotated = img_rgb.copy()
    if blur_bg:
        annotated = blur_background(annotated, result["box"], blur_strength=blur_strength)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (180, 180, 180), 2)
    cv2.putText(annotated, f"{emotion}  {confidence:.0%}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 2, cv2.LINE_AA)
    frame_placeholder.image(annotated, channels="RGB", use_container_width=True)
    status_placeholder.success(f"Detected: **{emotion}** ({confidence:.0%} confidence)")

    # Current mood card
    accent = EMOTION_COLORS.get(emotion, "#aaa")
    emotion_placeholder.markdown(f"""
    <div style="background:#1e1e1e; border:1px solid #2e2e2e; border-left:3px solid {accent};
                border-radius:8px; padding:16px 20px; display:flex; align-items:center; gap:16px">
      {emotion_icon_html(emotion)}
      <div>
        <div style="color:#888; font-size:0.8rem; margin-bottom:2px">Current mood</div>
        <div style="color:{accent}; font-size:1.6rem; font-weight:700">{emotion.capitalize()}</div>
        <div style="color:#555; font-size:0.8rem">{confidence:.0%} confidence</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Append to history (avoid duplicate consecutive entries)
    last = st.session_state.history[-1] if st.session_state.history else None
    if last is None or last["emotion"] != emotion:
        st.session_state.history.append({
            "emotion":    emotion,
            "confidence": confidence,
            "time":       datetime.now().strftime("%H:%M:%S"),
        })

    render_history()

# Webcam loop
if source == "Webcam":
    run = st.toggle("Start webcam", value=False)
    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Try 'Upload photo' instead.")
        else:
            stop_btn = st.button("Stop", icon=":material/stop:")
            last_detection = 0
            DETECTION_INTERVAL = 1.5

            last_box = None  # remember last known face box for continuous blur

            while not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                now = time.time()

                # Always apply blur using last known face box
                if blur_bg and last_box is not None:
                    display_frame = blur_background(frame_rgb, last_box, blur_strength=blur_strength)
                else:
                    display_frame = frame_rgb

                frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)

                if now - last_detection > DETECTION_INTERVAL:
                    result = detect_emotion(frame_rgb, confidence_threshold)
                    if result is not None:
                        last_box = result["box"]  # update box for next frames
                    process_frame(frame_rgb)
                    last_detection = now
                time.sleep(0.03)
            cap.release()

#  Upload photo
else:
    uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_rgb = np.array(img)
        frame_placeholder.image(img_rgb, channels="RGB", use_container_width=True)
        process_frame(img_rgb)
