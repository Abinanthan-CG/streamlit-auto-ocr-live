import streamlit as st
import cv2
import pytesseract
import numpy as np
from gtts import gTTS
import io
import time
import threading
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from streamlit_autorefresh import st_autorefresh

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SEEING WITH SOUND", page_icon="🎧", layout="wide")

# --- CUSTOM CSS FOR ACCESSIBILITY ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stAlert {
        background-color: #1e2130;
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #00ffcc !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if "last_ocr_time" not in st.session_state:
    st.session_state.last_ocr_time = time.time()
if "last_text" not in st.session_state:
    st.session_state.last_text = ""
if "last_objects" not in st.session_state:
    st.session_state.last_objects = []
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "snapshot_image" not in st.session_state:
    st.session_state.snapshot_image = None

# --- LOAD OBJECT DETECTION MODEL ---
@st.cache_resource
def load_model():
    prototxt = "deploy.prototxt"
    model = "mobilenet_iter_73000.caffemodel"
    net = cv2.dnn.readNetFromCaret(prototxt, model)
    return net

NET = load_model()
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
st.title("🎧 SEEING WITH SOUND")
st.subheader("Autonomous Visual-to-Auditory Aid System")

with st.expander("ℹ️ About the Project (from PDF)"):
    st.write("""
    **Objective:** Develop a visual speech aid system to assist visually impaired individuals in enhancing spatial awareness and navigation through intuitive auditory cues.
    
    **How it works:**
    1. **Capture Visual Data:** The system captures real-time video feed.
    2. **Process Visual Data:** Custom algorithms analyze the scene for objects and text.
    3. **Convert to Auditory Cues:** Interpreted data is transformed into clear voice announcements.
    4. **Output Sound:** Auditory cues are played to provide real-time environment feedback.
    """)

# --- LAYOUT ---
col_cam, col_info = st.columns([1.5, 1])

with col_cam:
    st.markdown("### 🎥 Live Environment Feed")
    camera_constraints = {
        "video": {"facingMode": "environment"}, 
        "audio": False
    }

    ctx = webrtc_streamer(
        key="see-with-sound",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints=camera_constraints,
        async_processing=True,
    )

with col_info:
    st.markdown("### 🛠️ Controls & Status")
    
    # Timer & Manual Trigger
    interval = 15 # Scans every 15 seconds for a more "real-time" feel
    
    if ctx.state.playing:
        st_autorefresh(interval=1000, key="timer_refresh")
        current_time = time.time()
        time_elapsed = current_time - st.session_state.last_ocr_time
        time_left = max(0, int(interval - time_elapsed))
        
        st.info(f"⏳ Next Scene Scan: **{time_left}s**")
        manual_trigger = st.button("📸 Sense Surroundings Now", type="primary")

        # --- PROCESSING LOGIC ---
        if time_elapsed >= interval or manual_trigger:
            if ctx.video_processor and ctx.video_processor.frame is not None:
                with ctx.video_processor.lock:
                    frame = ctx.video_processor.frame.copy()
                
                st.session_state.last_ocr_time = time.time()
                st.session_state.snapshot_image = frame

                with st.spinner("🧠 Analyzing Scene..."):
                    # 1. Object Detection
                    (h, w) = frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                    NET.setInput(blob)
                    detections = NET.forward()
                    
                    found_objects = []
                    for i in range(0, detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.5:
                            idx = int(detections[0, 0, i, 1])
                            found_objects.append(CLASSES[idx])
                    
                    st.session_state.last_objects = list(set(found_objects))

                    # 2. OCR (Text Recognition)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6').strip()
                    st.session_state.last_text = text if text else ""

                    # 3. Voice Feedback Generation
                    feedback_msg = ""
                    if st.session_state.last_objects:
                        feedback_msg += "I see " + ", ".join(st.session_state.last_objects) + ". "
                    
                    if st.session_state.last_text:
                        feedback_msg += "The text says: " + st.session_state.last_text
                    
                    if not feedback_msg:
                        feedback_msg = "Nothing significant detected."

                    tts = gTTS(text=feedback_msg, lang='en')
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    st.session_state.last_audio = fp.read()

# --- RESULTS SECTION ---
st.markdown("---")
res_col1, res_col2 = st.columns([1, 1])

with res_col1:
    if st.session_state.snapshot_image is not None:
        st.image(st.session_state.snapshot_image, channels="BGR", caption="Last Captured Scene")

with res_col2:
    st.markdown("### 🔊 Auditory Feedback")
    if st.session_state.last_audio:
        st.audio(st.session_state.last_audio, format="audio/mp3", autoplay=True)
    
    st.markdown("### 📝 Scene Intelligence")
    if st.session_state.last_objects:
        st.success(f"**Objects Detected:** {', '.join(st.session_state.last_objects)}")
    else:
        st.info("No objects detected in last scan.")
    
    if st.session_state.last_text:
        st.warning(f"**Text Recognized:** {st.session_state.last_text}")
    else:
        st.info("No text recognized in last scan.")

# --- FOOTER ---
st.markdown("---")
st.caption("SEEING WITH SOUND - Developed to assist and empower the visually impaired.")
