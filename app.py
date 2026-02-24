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
st.set_page_config(page_title="Live Auto OCR", page_icon="⏱️", layout="centered")

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

# --- UI ---
st.title("⏱️ 30-Second Auto Reader")
st.markdown("1. Click **START** below.\n2. Allow camera access.\n3. The AI will read text every 30 seconds.")

ctx = webrtc_streamer(
    key="ocr-streamer",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.state.playing:
    # Refresh UI every 1 second to update timer
    st_autorefresh(interval=1000, key="timer_refresh")
    
    current_time = time.time()
    time_elapsed = current_time - st.session_state.last_ocr_time
    # 30 second interval
    interval = 30
    time_left = max(0, int(interval - time_elapsed))
    
    st.info(f"⏳ Next Scan in: **{time_left}s**")
    
    # TRIGGER OCR
    if time_elapsed >= interval:
        if ctx.video_processor and ctx.video_processor.frame is not None:
            with ctx.video_processor.lock:
                frame = ctx.video_processor.frame.copy()
            
            # Reset timer immediately
            st.session_state.last_ocr_time = time.time()

            with st.spinner("👀 Reading text..."):
                # Preprocessing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Simple thresholding often works better for documents than adaptive in variable light
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # OCR
                text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6').strip()
                
                st.image(frame, channels="BGR", caption="Snapshot Processed")
                
                if text:
                    st.success(f"**Read:** {text}")
                    # TTS
                    tts = gTTS(text=text, lang='en')
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    autoplay_audio(fp.read())
                else:
                    st.warning("No text detected.")