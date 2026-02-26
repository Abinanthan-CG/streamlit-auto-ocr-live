import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
import io
import time
import threading
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from streamlit_autorefresh import st_autorefresh

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SEEING WITH SOUND: Navigator", page_icon="🧭", layout="wide")

# --- PREMIUM SONIC UI & CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    .main {
        background-color: #05070a;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 2px;
        color: #00f2ff !important;
        text-shadow: 0 0 10px rgba(0, 242, 255, 0.5);
    }
    
    .stAlert {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        color: white;
    }
    
    .radar-container {
        position: relative;
        width: 100%;
        height: 300px;
        background: radial-gradient(circle, #0a141a 0%, #05070a 100%);
        border: 2px solid #1a3a4a;
        border-radius: 50% 50% 10px 10px;
        overflow: hidden;
        margin-bottom: 20px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .radar-sweep {
        position: absolute;
        width: 100%;
        height: 100%;
        background: conic-gradient(from 0deg, transparent 0deg, rgba(0, 242, 255, 0.2) 30deg, transparent 31deg);
        animation: rotate 4s linear infinite;
        border-radius: 50%;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .status-panel {
        background: rgba(0, 242, 255, 0.03);
        border-right: 4px solid #00f2ff;
        padding: 20px;
        border-radius: 10px;
    }
    
    .nav-command {
        font-size: 24px;
        font-weight: bold;
        color: #ff3e3e;
        text-align: center;
        padding: 10px;
        border-top: 2px dashed #ff3e3e;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = time.time()
if "last_objects" not in st.session_state:
    st.session_state.last_objects = [] # List of dicts: {name, zone, proximity}
if "nav_command" not in st.session_state:
    st.session_state.nav_command = "SYSTEM READY"
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "snapshot_image" not in st.session_state:
    st.session_state.snapshot_image = None

# --- LOAD OBJECT DETECTION MODEL ---
@st.cache_resource
def load_navigator_model():
    prototxt = "deploy.prototxt"
    model = "mobilenet_iter_73000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    return net

NET = load_navigator_model()
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# --- WEBRTC CONFIGURATION ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- VIDEO PROCESSOR ---
class VideoProcessor:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            self.frame = img
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI HEADER ---
st.title("🧭 SEEING WITH SOUND: NAVIGATOR")
st.markdown("---")

# --- MAIN LAYOUT ---
col_side, col_center, col_radar = st.columns([0.8, 1.5, 0.8])

with col_side:
    st.markdown("<div class='status-panel'>", unsafe_allow_html=True)
    st.markdown("### 📊 Status")
    st.metric("System Load", "Optimal", delta="0ms")
    st.metric("Detection Confidence", "0.50 (High)")
    
    st.markdown("### ⚙️ Guidance")
    interval = st.slider("Update Frequency (s)", 3, 30, 8)
    st.write(f"Scanning environment every {interval}s")
    st.markdown("</div>", unsafe_allow_html=True)

with col_center:
    camera_constraints = {"video": {"facingMode": "environment"}, "audio": False}
    ctx = webrtc_streamer(
        key="navigator-streamer",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints=camera_constraints,
        async_processing=True,
    )
    
    if st.session_state.snapshot_image is not None:
        st.image(st.session_state.snapshot_image, channels="BGR", use_container_width=True)
    else:
        st.info("Start the camera to begin navigation.")

with col_radar:
    st.markdown("### 🛰️ Spatial Radar")
    st.markdown("""
        <div class='radar-container'>
            <div class='radar-sweep'></div>
            <div style='z-index: 10; color: #00f2ff; font-family: Orbitron; font-size: 10px;'>SONIC SWEEP ACTIVE</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation Dashboard
    if st.session_state.nav_command != "SYSTEM READY":
        st.error(f"**CMD: {st.session_state.nav_command}**")
        if st.session_state.last_audio:
            st.audio(st.session_state.last_audio, format="audio/mp3", autoplay=True)
    else:
        st.success("PATH CLEAR")

# --- PROCESSING LOGIC ---
if ctx.state.playing:
    st_autorefresh(interval=1000, key="nav_refresh")
    current_time = time.time()
    time_elapsed = current_time - st.session_state.last_scan_time

    if time_elapsed >= interval:
        if ctx.video_processor and ctx.video_processor.frame is not None:
            with ctx.video_processor.lock:
                frame = ctx.video_processor.frame.copy()
            
            st.session_state.last_scan_time = time.time()
            (h, w) = frame.shape[:2]
            
            # Neural Net Inference
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            NET.setInput(blob)
            detections = NET.forward()
            
            current_objects = []
            navigation_advice = ""
            collision_warning = False
            
            # Define Zones: Left (0-33%), Center (33-66%), Right (66-100%)
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1])
                    label = CLASSES[idx]
                    
                    # Get Bounding Box
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    centerX = (startX + endX) / 2
                    
                    # Spatial Logic
                    zone = "Center"
                    if centerX < w/3: zone = "Left"
                    elif centerX > 2*w/3: zone = "Right"
                    
                    # Proximity (Area as Proxy)
                    area_percent = ((endX - startX) * (endY - startY)) / (w * h)
                    proximity = "Far"
                    if area_percent > 0.15: proximity = "Near"
                    if area_percent > 0.4: proximity = "CRITICAL"
                    
                    current_objects.append({"name": label, "zone": zone, "proximity": proximity})
                    
                    # Annotate Frame
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 242, 255), 2)
                    cv2.putText(frame, f"{label} ({proximity})", (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 242, 255), 2)

            # --- DECISION ENGINE ---
            st.session_state.last_objects = current_objects
            st.session_state.snapshot_image = frame
            
            center_obstructions = [obj for obj in current_objects if obj['zone'] == "Center"]
            critical_obstructions = [obj for obj in current_objects if obj['proximity'] == "CRITICAL"]
            
            if critical_obstructions:
                navigation_advice = "STOP! Object directly in front."
                st.session_state.nav_command = "🛑 STOP IMMEDIATELY"
            elif center_obstructions:
                # Check which side is clearer
                left_objs = len([obj for obj in current_objects if obj['zone'] == "Left"])
                right_objs = len([obj for obj in current_objects if obj['zone'] == "Right"])
                
                if left_objs <= right_objs:
                    navigation_advice = f"Object ahead. Move slightly to your left."
                    st.session_state.nav_command = "⬅️ MOVE LEFT"
                else:
                    navigation_advice = f"Object ahead. Move slightly to your right."
                    st.session_state.nav_command = "➡️ MOVE RIGHT"
            elif current_objects:
                obj_names = list(set([obj['name'] for obj in current_objects]))
                navigation_advice = f"I see {', '.join(obj_names)}. Path looks clear."
                st.session_state.nav_command = "✅ PATH CLEAR"
            else:
                navigation_advice = "Environment clear."
                st.session_state.nav_command = "✅ PATH CLEAR"

            # Generate Voice
            tts = gTTS(text=navigation_advice, lang='en')
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            st.session_state.last_audio = fp.read()
            st.rerun()

# --- FOOTER ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #1a3a4a;'>SEEING WITH SOUND: Navigator Edition | Powered by Neural Spatial Awareness</div>", unsafe_allow_html=True)
