import streamlit as st
import cv2
import numpy as np
import time
import threading
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components
from ultralytics import YOLO

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
if "msg" not in st.session_state: st.session_state.msg = "SCENE INTERPRETER INITIALIZING"
if "msg_type" not in st.session_state: st.session_state.msg_type = "clear"
if "snapshot" not in st.session_state: st.session_state.snapshot = None
if "speech_queue" not in st.session_state: st.session_state.speech_queue = None

# --- ENGINE: YOLOv8 ---
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt") # Nano for speed

MODEL = load_yolo()

# Priority rankings (1 = Critical Moving, 2 = Fixed Obstacle, 3 = Context)
PRIORITIES = {
    # Critical Moving Hazards (Tier 1)
    "person": 1, "car": 1, "bus": 1, "truck": 1, "bicycle": 1, "motorcycle": 1, "dog": 1, "cat": 1,
    # Fixed Navigation Obstacles (Tier 2)
    "bench": 2, "chair": 2, "couch": 2, "potted plant": 2, "stairs": 2, "door": 2, "backpack": 2, "umbrella": 2,
    "handbag": 2, "suitcase": 2, "dining table": 2, "bed": 2, "toilet": 2, "tv": 2, "laptop": 2, "mouse": 2,
    "remote": 2, "keyboard": 2, "cell phone": 2, "microwave": 2, "oven": 2, "toaster": 2, "sink": 2, 
    "refrigerator": 2, "book": 2, "clock": 2, "vase": 2, "scissors": 2, "teddy bear": 2, "hair drier": 2, "toothbrush": 2
}

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
    **SEEING WITH SOUND: Advanced Scene Understanding**
    - **YOLOv8 Engine:** Detecting 80+ objects with 90% higher accuracy.
    - **Scene Interpreter:** Prioritizes moving hazards over fixed obstacles.
    - **Turbo Audio:** Instant verbal narrative for navigation context.
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
    st.session_state.speech_queue = None 

# --- LOGIC: SCENE INTERPRETER ---
if ctx.state.playing:
    st_autorefresh(interval=scan_interval * 1000, key="nav_loop")
    if ctx.video_processor and ctx.video_processor.frame is not None:
        with ctx.video_processor.lock: frame = ctx.video_processor.frame.copy()
        
        # Inference
        results = MODEL(frame, stream=True, conf=0.45)
        objs = []
        h, w = frame.shape[:2]
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                label = MODEL.names[cls]
                conf = float(box.conf[0])
                
                # Logic calc
                cx = (x1 + x2) / 2
                area = ((x2 - x1) * (y2 - y1)) / (w * h)
                zone = "center" if w/3 < cx < 2*w/3 else ("left" if cx < w/3 else "right")
                priority = PRIORITIES.get(label, 3)
                
                objs.append({"label": label, "zone": zone, "area": area, "prio": priority, "coords": (int(x1), int(y1), int(x2), int(y2))})
                
                # Draw boxes
                color = (0, 255, 0) if area < 0.3 else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label.upper()} {int(conf*100)}%", (int(x1), int(y1)-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # -- SCENE INTERPRETATION --
        msg, mtype, speech = "PATH CLEAR", "clear", "The path ahead is clear."
        
        # Sort by priority and size
        objs_sorted = sorted(objs, key=lambda x: (x['prio'], -x['area']))
        
        if objs:
            primary = objs_sorted[0]
            label = primary['label']
            zone = primary['zone']
            
            # Contextual Narrative
            if primary['area'] > 0.4:
                msg, mtype, speech = f"STOP! {label.upper()}", "stop", f"Stop immediately. A {label} is directly in front of you."
            elif zone == "center":
                left_objs = [o for o in objs if o['zone'] == 'left']
                right_objs = [o for o in objs if o['zone'] == 'right']
                
                if not left_objs:
                    msg, mtype, speech = f"MOVE LEFT", "stop", f"A {label} is blocking the center. Move slightly to your left."
                elif not right_objs:
                    msg, mtype, speech = f"MOVE RIGHT", "stop", f"A {label} is blocking the center. Move slightly to your right."
                else:
                    msg, mtype, speech = f"OBSTACLE AHEAD", "stop", f"The center is blocked by a {label}, and the sides are also congested."
            else:
                speech_text = f"I see a {label} on your {zone}."
                if len(objs) > 1:
                    secondary = objs_sorted[1]
                    speech_text += f" There is also a {secondary['label']} nearby."
                msg, mtype, speech = f"{label.upper()} {zone}", "warn", speech_text

        st.session_state.msg = msg
        st.session_state.msg_type = mtype
        st.session_state.snapshot = frame
        st.session_state.speech_queue = speech
        st.rerun()
