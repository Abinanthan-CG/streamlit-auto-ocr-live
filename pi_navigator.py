import cv2
import numpy as np
import time
import threading
import pyttsx3
import argparse
from picamera2 import Picamera2

# --- CONFIGURATION ---
MODEL_PROTOTXT = "deploy.prototxt"
MODEL_CAFFE = "mobilenet_iter_73000.caffemodel"
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

class NavigatorPi:
    def __init__(self, frequency=3.0, display=False):
        self.frequency = frequency
        self.display = display
        
        # Initialize TTS Engine (espeak-ng usually works best on Pi)
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160) # Slow enough for clarity
        
        # Load Model
        print("[INFO] Loading detection model...")
        self.net = cv2.dnn.readNetFromCaffe(MODEL_PROTOTXT, MODEL_CAFFE)
        
        # Initialize Camera (Picamera2 for RPi 5)
        print("[INFO] Initializing Picamera2...")
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_video_configuration(main={"size": (640, 480)}))
        self.picam2.start()
        
        self.last_speech_time = 0
        self.lock = threading.Lock()
        self.running = True

    def speak(self, text):
        """Threaded speech to avoid blocking the vision loop."""
        def _say():
            with self.lock:
                self.engine.say(text)
                self.engine.runAndWait()
        threading.Thread(target=_say).start()

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        objs = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (sx, sy, ex, ey) = box.astype("int")
                area = ((ex-sx)*(ey-sy)) / (w*h)
                cx = (sx+ex)/2
                zone = "center" if w/3 < cx < 2*w/3 else ("left" if cx < w/3 else "right")
                label = CLASSES[int(detections[0, 0, i, 1])]
                objs.append({"label": label, "zone": zone, "area": area, "box": (sx, sy, ex, ey)})
        return objs

    def run(self):
        print("[INFO] Starting Navigator Loop...")
        try:
            while self.running:
                # Capture frame
                frame = self.picam2.capture_array()
                
                # Check timing
                current_time = time.time()
                if current_time - self.last_speech_time >= self.frequency:
                    objs = self.process_frame(frame)
                    
                    # Logic: Decision Engine
                    msg, speech = "PATH CLEAR", "Path is clear."
                    center = [o for o in objs if o['zone'] == 'center']
                    critical = [o for o in objs if o['area'] > 0.35]
                    
                    if critical:
                        speech = "Stop immediately. Object directly in front."
                    elif center:
                        left_clear = len([o for o in objs if o['zone'] == 'left']) == 0
                        dir_hint = "Move left." if left_clear else "Move right."
                        speech = f"Obstacle ahead. {dir_hint}"
                    elif objs:
                        names = list(set([o['label'] for o in objs]))
                        speech = f"I see a {names[0]}. Path looks okay."
                    
                    print(f"[NAV] {speech}")
                    self.speak(speech)
                    self.last_speech_time = current_time

                    if self.display:
                        for o in objs:
                            sx, sy, ex, ey = o['box']
                            cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)
                            cv2.putText(frame, o['label'], (sx, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if self.display:
                    cv2.imshow("SEEING WITH SOUND - RPi Edition", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except KeyboardInterrupt:
            print("\n[INFO] Stopping...")
        finally:
            self.picam2.stop()
            self.picam2.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEEING WITH SOUND - Pi 5 Edition")
    parser.add_argument("--freq", type=float, default=3.0, help="Scan frequency in seconds")
    parser.add_argument("--display", action="store_true", help="Show video window")
    args = parser.parse_args()

    nav = NavigatorPi(frequency=args.freq, display=args.display)
    nav.run()
