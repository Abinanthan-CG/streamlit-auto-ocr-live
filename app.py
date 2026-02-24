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
# STUN servers allow the camera to work over the internet/mobile data
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- VIDEO PROCESSOR ---
class VideoProcessor:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()

    def recv(self, frame):
        # Convert the WebRTC frame to an OpenCV image (numpy array)
        img = frame.to_ndarray(format="bgr24")
        
        # Save the latest frame to memory safely
        with self.lock:
            self.frame = img
            
        # Return the frame so the user sees the video feed
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- AUDIO AUTOPLAY FUNCTION ---
def autoplay_audio(audio_bytes):
    """
    Injects HTML to play audio automatically on mobile/desktop 
    without requiring a click.
    """
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
        <audio autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

# --- UI ---
st.title("📸 AI Mobile Scanner")
st.markdown("""
**Instructions:**
1. Click **START** below.
2. If asked, allow camera access.
3. Point your **back camera** at text.
4. Every **30 seconds**, the AI will read it to you.
""")

# --- THE MAGIC SETTING FOR BACK CAMERA ---
# "facingMode": "environment" tells the phone to use the rear camera.
camera_constraints = {
    "video": {"facingMode": "environment"},
    "audio": False
}

ctx = webrtc_streamer(
    key="ocr-streamer",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints=camera_constraints, # <--- THIS ENABLES BACK CAMERA
    async_processing=True,
)

# --- MAIN AUTOMATION LOOP ---
if ctx.state.playing:
    # Refresh the UI every 1 second to update the countdown timer
    st_autorefresh(interval=1000, key="timer_refresh")
    
    current_time = time.time()
    time_elapsed = current_time - st.session_state.last_ocr_time
    interval = 30 # Time in seconds between scans
    time_left = max(0, int(interval - time_elapsed))
    
    # Display Countdown
    st.info(f"⏳ Next AI Scan in: **{time_left}s**")
    
    # TRIGGER OCR WHEN TIMER HITS 30s
    if time_elapsed >= interval:
        if ctx.video_processor and ctx.video_processor.frame is not None:
            # Grab the latest frame thread-safely
            with ctx.video_processor.lock:
                frame = ctx.video_processor.frame.copy()
            
            # Reset timer immediately so the loop continues
            st.session_state.last_ocr_time = time.time()

            with st.spinner("👀 Reading text..."):
                # 1. Preprocessing for Mobile Cameras (often clearer but shaky)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Use OTSU thresholding (Better for documents with varying lighting)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # 2. Run OCR
                # psm 6 = Assume a single uniform block of text
                text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6').strip()
                
                # Show the snapshot that was analyzed
                st.image(frame, channels="BGR", caption="Snapshot Processed")
                
                if text:
                    st.success(f"**Read:** {text}")
                    
                    # 3. Generate Audio
                    tts = gTTS(text=text, lang='en')
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    
                    # 4. Play Audio
                    autoplay_audio(fp.read())
                else:
                    st.warning("No text detected. Try holding the camera steady.")
