import streamlit as st
import cv2
import numpy as np
import time
import threading
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

# --- SMART MINIMALIST UI CONFIG ---
st.set_page_config(page_title="NAVIGATOR", page_icon="🧭", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    .main { background-color: #000000; color: #ffffff; font-family: 'Inter', sans-serif; }
    .stApp { max-width: 100%; padding: 0; }
    
    .stExpander {
        background: rgba(30, 30, 30, 0.5);
        border: 1px solid #333;
        border-radius: 10px;
        margin: 10px;
    }
    
    .element-container img { border-radius: 0px; width: 100% !important; }
    
    .nav-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: rgba(20, 20, 20, 0.9);
        backdrop-filter: blur(10px);
        padding: 20px;
        text-align: center;
        border-top: 2px solid #333;
        z-index: 1000;
        margin-bottom: 0;
    }
    
    .status-text {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .status-clear { color: #00ff88; }
    .status-warn { color: #ffcc00; }
    .status-stop { color: #ff3333; text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

# --- WEB SPEECH API HELPER ---
def speak(text):
    if text:
        components.html(f"""
            <script>
            var msg = new SpeechSynthesisUtterance('{text}');
            window.speechSynthesis.cancel(); // Stop any current speech
            window.speechSynthesis.speak(msg);
            </script>
        """, height=0)

# --- SESSION STATE ---
if "last_scan" not in st.session_state: st.session_state.last_scan = time.time()
if "msg" not in st.session_state: st.session_state.msg = "SYSTEM INITIALIZING"
if "msg_type" not in st.session_state: st.session_state.msg_type = "clear"
if "snapshot" not in st.session_state: st.session_state.snapshot = None
if "speech_queue" not in st.session_state: st.session_state.speech_queue = None

# --- MODEL ---
@st.cache_resource
def load_model():
    return cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

NET = load_model()
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# --- CAMERA ---
class Processor:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with self.lock: self.frame = img
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- SMART MINIMALIST TOP PANEL ---
with st.expander("⚙️ Settings & Info"):
    scan_interval = st.slider("Scan Frequency (seconds)", 3, 30, 5)
    st.markdown("""
    **SEEING WITH SOUND: Navigator**
    1. **Senses:** Automatically scans the environment for obstacles.
    2. **Maps:** Detects objects in Left, Center, and Right zones.
    3. **Guides:** Provides instant voice commands to help you navigate safely.
    """)

ctx = webrtc_streamer(
    key="nav",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_processor_factory=Processor,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    async_processing=True,
)

# --- UI DISPLAY ---
if st.session_state.snapshot is not None:
    st.image(st.session_state.snapshot, channels="BGR", use_container_width=True)
else:
    st.markdown("<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666;'>CAMERA READY</div>", unsafe_allow_html=True)

# Navigation Alert Bar
st.markdown(f"""
    <div class="nav-bar">
        <div class="status-text status-{st.session_state.msg_type}">
            {st.session_state.msg}
        </div>
    </div>
""", unsafe_allow_html=True)

# Instant Speech Trigger
if st.session_state.speech_queue:
    speak(st.session_state.speech_queue)
    st.session_state.speech_queue = None # Clear after triggering

# --- LOGIC ---
if ctx.state.playing:
    st_autorefresh(interval=scan_interval * 1000, key="nav_loop")
    if ctx.video_processor and ctx.video_processor.frame is not None:
        with ctx.video_processor.lock: frame = ctx.video_processor.frame.copy()
        
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        NET.setInput(blob)
        detections = NET.forward()
        
        objs = []
        for i in range(detections.shape[2]):
            if detections[0, 0, i, 2] > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (sx, sy, ex, ey) = box.astype("int")
                area = ((ex-sx)*(ey-sy)) / (w*h)
                cx = (sx+ex)/2
                zone = "center" if w/3 < cx < 2*w/3 else ("left" if cx < w/3 else "right")
                objs.append({"label": CLASSES[int(detections[0, 0, i, 1])], "zone": zone, "area": area})
        
        # Determine Command
        msg, mtype, speech = "PATH CLEAR", "clear", "Path is clear."
        center = [o for o in objs if o['zone'] == 'center']
        critical = [o for o in objs if o['area'] > 0.35]
        
        if critical:
            msg, mtype, speech = "STOP! DANGER", "stop", "Stop immediately. Object in front."
        elif center:
            left_clear = len([o for o in objs if o['zone'] == 'left']) == 0
            command = "MOVE LEFT" if left_clear else "MOVE RIGHT"
            msg, mtype, speech = command, "stop", f"Obstacle ahead. {command.lower()}."
        elif objs:
            names = list(set([o['label'] for o in objs]))
            msg, mtype, speech = f"CLEAR - {names[0].upper()}", "warn", f"I see a {names[0]}. Path looks okay."
        
        st.session_state.msg = msg
        st.session_state.msg_type = mtype
        st.session_state.snapshot = frame
        st.session_state.speech_queue = speech # Queue for immediate browser speech
        
        st.rerun()
