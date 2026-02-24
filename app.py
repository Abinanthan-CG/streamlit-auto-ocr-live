import streamlit as st
import cv2
import pytesseract
import numpy as np
from gtts import gTTS
import io
import base64
import time
import threading
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from streamlit_autorefresh import st_autorefresh

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Live Auto OCR", page_icon="📸", layout="centered")

# --- INITIALIZE STATE ---
if "last_ocr_time" not in st.session_state:
    st.session_state.last_ocr_time = time.time()

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
        # Convert WebRTC frame to OpenCV image
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            self.frame = img
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- AUDIO AUTOPLAY FUNCTION ---
def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
        <audio autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

# --- UI HEADER ---
st.title("📸 AI Mobile Scanner")
st.markdown("Point your back camera at text. It reads every 30s, or when you click **Snap**.")

# --- CAMERA SETTINGS (Back Camera) ---
camera_constraints = {
    "video": {"facingMode": "environment"}, # Forces back camera on mobile
    "audio": False
}

# --- STREAMER COMPONENT ---
ctx = webrtc_streamer(
    key="ocr-streamer",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints=camera_constraints,
    async_processing=True,
)

# --- MAIN LOGIC LOOP ---
if ctx.state.playing:
    # 1. Automatic Refresh (Keeps the UI alive and updating the timer)
    st_autorefresh(interval=1000, key="timer_refresh")
    
    # 2. Calculate Timer
    current_time = time.time()
    time_elapsed = current_time - st.session_state.last_ocr_time
    interval = 30
    time_left = max(0, int(interval - time_elapsed))
    
    # 3. UI Controls
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(f"⏳ Next Auto-Scan in: **{time_left}s**")
    with col2:
        # THE MANUAL BUTTON
        manual_trigger = st.button("📸 Snap Now", type="primary", use_container_width=True)

    # 4. Trigger Logic: Run if Timer is up OR Manual Button is clicked
    if time_elapsed >= interval or manual_trigger:
        
        # Check if camera is ready
        if ctx.video_processor and ctx.video_processor.frame is not None:
            
            # Grab frame thread-safely
            with ctx.video_processor.lock:
                frame = ctx.video_processor.frame.copy()
            
            # IMPORTANT: Reset the timer immediately!
            st.session_state.last_ocr_time = time.time()

            with st.spinner("👀 Processing..."):
                # --- PREPROCESSING ---
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # OTSU Thresholding (Good for documents)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # --- OCR ---
                text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6').strip()
                
                # Show result
                st.image(frame, channels="BGR", caption="Snapshot Processed")
                
                if text:
                    st.success(f"**Read:** {text}")
                    
                    # --- TTS AUDIO ---
                    tts = gTTS(text=text, lang='en')
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    
                    # Autoplay
                    autoplay_audio(fp.read())
                else:
                    st.warning("No text detected.")
        else:
            if manual_trigger:
                st.error("Camera not ready yet. Wait a second!")
