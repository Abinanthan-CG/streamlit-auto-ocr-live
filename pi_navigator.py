import cv2
import numpy as np
import time
import threading
import pyttsx3
import argparse
import os
from picamera2 import Picamera2
from ultralytics import YOLO

# --- CONFIGURATION ---
PRIORITIES = {
    "person": 1, "car": 1, "bus": 1, "truck": 1, "bicycle": 1, "motorcycle": 1, "dog": 1, "cat": 1,
    "bench": 2, "chair": 2, "couch": 2, "potted plant": 2, "stairs": 2, "door": 2, "backpack": 2, "umbrella": 2,
    "handbag": 2, "suitcase": 2, "dining table": 2, "bed": 2, "toilet": 2, "tv": 2, "laptop": 2, "mouse": 2,
    "remote": 2, "keyboard": 2, "cell phone": 2, "microwave": 2, "oven": 2, "toaster": 2, "sink": 2, 
    "refrigerator": 2, "book": 2, "clock": 2, "vase": 2, "scissors": 2, "teddy bear": 2, "hair drier": 2, "toothbrush": 2
}

class NavigatorPi:
    def __init__(self, frequency=1.0, display=False):
        self.frequency = frequency
        self.display = display
        
        # Initialize TTS Engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        
        # Load Model (Prioritize OpenVINO for RPi 5)
        ov_model = "yolov8n_openvino_model"
        if os.path.exists(ov_model):
            print(f"[INFO] Using ACCELERATED OpenVINO engine: {ov_model}")
            self.model = YOLO(ov_model, task="detect")
        else:
            print("[INFO] OpenVINO model not found. Falling back to standard YOLOv8-nano engine...")
            self.model = YOLO("yolov8n.pt")
        
        # Initialize Camera
        print("[INFO] Initializing Picamera2...")
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_video_configuration(main={"size": (640, 480)}))
        self.picam2.start()
        
        self.last_speech_time = 0
        self.lock = threading.Lock()
        self.running = True

    def speak(self, text):
        def _say():
            with self.lock:
                self.engine.say(text)
                self.engine.runAndWait()
        threading.Thread(target=_say).start()

    def run(self):
        print(f"[INFO] Starting Navigator Loop (Scan Interval: {self.frequency}s)...")
        try:
            while self.running:
                frame = self.picam2.capture_array()
                h, w = frame.shape[:2]
                
                current_time = time.time()
                # Run inference as fast as possible for smooth display, but respect speech frequency
                results = self.model(frame, verbose=False, conf=0.45)
                objs = []
                
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cls = int(box.cls[0])
                        label = self.model.names[cls]
                        area = ((x2 - x1) * (y2 - y1)) / (w * h)
                        cx = (x1 + x2) / 2
                        zone = "center" if w/3 < cx < 2*w/3 else ("left" if cx < w/3 else "right")
                        priority = PRIORITIES.get(label, 3)
                        objs.append({"label": label, "zone": zone, "area": area, "prio": priority, "coords": (int(x1), int(y1), int(x2), int(y2))})

                # Speech Logic (Every N seconds)
                if current_time - self.last_speech_time >= self.frequency:
                    msg, speech = "PATH CLEAR", "The path ahead is clear."
                    objs_sorted = sorted(objs, key=lambda x: (x['prio'], -x['area']))
                    
                    if objs:
                        primary = objs_sorted[0]
                        label = primary['label']
                        zone = primary['zone']
                        
                        if primary['area'] > 0.4:
                            speech = f"Stop immediately. A {label} is directly in front of you."
                        elif zone == "center":
                            left_clear = len([o for o in objs if o['zone'] == 'left']) == 0
                            dir_hint = "Move left." if left_clear else "Move right."
                            speech = f"A {label} is blocking the center. {dir_hint}"
                        else:
                            speech = f"I see a {label} on your {zone}."
                    
                    print(f"[NAV] {speech}")
                    self.speak(speech)
                    self.last_speech_time = current_time

                # Display Logic
                if self.display:
                    for o in objs:
                        x1, y1, x2, y2 = o['coords']
                        color = (0, 255, 0) if o['area'] < 0.4 else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, o['label'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    cv2.imshow("SEEING WITH SOUND - Accelerated RPi 5", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            self.picam2.stop()
            self.picam2.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEEING WITH SOUND - Accelerated Pi 5")
    parser.add_argument("--freq", type=float, default=1.0, help="Scan frequency in seconds")
    parser.add_argument("--display", action="store_true", help="Show video window")
    args = parser.parse_args()

    nav = NavigatorPi(frequency=args.freq, display=args.display)
    nav.run()
