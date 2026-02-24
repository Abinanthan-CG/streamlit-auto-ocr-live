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
st.set_page_config(page_title="Live Auto OCR", page_icon="🎧", layout="centered")

# --- INITIALIZE SESSION STATE ---
# We need to store the audio/text so it persists between screen refreshes
if "last_ocr_time" not in st.session_state:
    st.session_state.last_ocr_time = time.time()
if "last_text" not in st.session_state:
    st.session_state.last_text = ""
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "snapshot_image" not in st.session_state:
    st.session_state.snapshot_image = None

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
st.title("🎧 AI Scanner (Manual Audio)")
st.markdown("Point camera at text. It scans every 30s. **Click Play** to hear the result.")

# --- CAMERA SETUP (Back Camera) ---
camera_constraints = {
    "video": {"facingMode": "environment"}, 
    "audio": False
}

ctx = webrtc_streamer(
    key="ocr-streamer",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints=camera_constraints,
    async_processing=True,
)

# --- MAIN LOOP ---
if ctx.state.playing:
    # 1. Refresh UI every 1 second (1000ms) for the timer
    st_autorefresh(interval=1000, key="timer_refresh")
    
    # 2. Timer Logic
    current_time = time.time()
    time_elapsed = current_time - st.session_state.last_ocr_time
    interval = 30
    time_left = max(0, int(interval - time_elapsed))
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(f"⏳ Next Scan in: **{time_left}s**")
    with col2:
        manual_trigger = st.button("📸 Snap Now", type="primary", use_container_width=True)

    # 3. TRIGGER LOGIC (Timer OR Button)
    if time_elapsed >= interval or manual_trigger:
        if ctx.video_processor and ctx.video_processor.frame is not None:
            # Capture Frame
            with ctx.video_processor.lock:
                frame = ctx.video_processor.frame.copy()
            
            # Reset Timer
            st.session_state.last_ocr_time = time.time()

            # --- PROCESSING ---
            with st.spinner("👀 Processing..."):
                # Preprocess
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # OTSU Thresholding (Best for text)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # OCR
                text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6').strip()
                
                # Update Session State (This saves the result to memory!)
                st.session_state.snapshot_image = frame
                
                if text:
                    st.session_state.last_text = text
                    
                    # Generate Audio
                    tts = gTTS(text=text, lang='en')
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    st.session_state.last_audio = fp.read()
                else:
                    st.session_state.last_text = "No text detected."
                    st.session_state.last_audio = None

    # 4. PERSISTENT DISPLAY SECTION
    # This part runs every second, ensuring the player stays visible
    st.markdown("---")
    
    if st.session_state.snapshot_image is not None:
        st.image(st.session_state.snapshot_image, channels="BGR", caption="Last Snapshot", width=300)

    if st.session_state.last_text:
        if st.session_state.last_text == "No text detected.":
            st.warning(st.session_state.last_text)
        else:
            st.success("**Extracted Text:**")
            st.write(st.session_state.last_text)

            # --- THE MANUAL AUDIO PLAYER ---
            if st.session_state.last_audio:
                st.markdown("### 🔊 Audio Result")
                st.audio(st.session_state.last_audio, format="audio/mp3")
