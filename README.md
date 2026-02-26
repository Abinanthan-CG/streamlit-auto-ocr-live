# 🧭 SEEING WITH SOUND: Autonomous Navigator

```text
   _____ ______ ______ _____ _   _  _____  __          _______ _______ _    _
  / ____|  ____|  ____|_   _| \ | |/ ____| \ \        / /_   _|__   __| |  | |
 | (___ | |__  | |__    | | |  \| | |  __   \ \  /\  / /  | |    | |  | |__| |
  \___ \|  __| |  __|   | | | . ` | | |_ |   \ \/  \/ /   | |    | |  |  __  |
  ____) | |____| |____ _| |_| |\  | |__| |    \  /\  /   _| |_   | |  | |  | |
 |_____/|______|______|_____|_| \_|\_____|     \/  \/   |_____|  |_|  |_|  |_|

                   --- VISUAL-TO-AUDITORY AID SYSTEM ---
```

**SEEING WITH SOUND** is a state-of-the-art navigation assistant designed to empower visually impaired individuals. By converting real-time environmental data into intuitive spatial audio cues, the system helps users avoid obstacles and navigate their surroundings with confidence.

---

## 🚀 Versions

### 🌐 1. Web & Mobile Edition (Portable)

Designed for smartphones and laptops. Access it through any modern browser to transform your phone into a wearable navigator.

- **Turbo Audio:** Uses Web Speech API for near-zero voice latency.
- **Smart Minimalist UI:** Zero-clutter, high-contrast mobile interface.
- **Responsive:** Adapts perfectly to portrait or landscape mobile screens.

### 🍓 2. Raspberry Pi 5 Edition (Hardware)

A standalone, offline version designed for independent hardware setups (e.g., smart glasses or handheld devices).

- **Picamera2 Optimized:** High-speed processing for RPi 5.
- **Offline TTS:** Works without an internet connection using `espeak-ng`.
- **Headless Ready:** Can be configured to launch automatically on boot.

---

## 🛠️ Key Features

- **Spatial Mapping:** Categorizes surroundings into **Left, Center,** and **Right** zones.
- **Collision Avoidance:** Detects objects in the "Center" zone and calculates the safest clearing path.
- **Proximity Alerts:** Issues high-priority **"STOP"** commands for critically close obstructions.
- **Instant Verbal Guidance:** Directional instructions ("Move Left", "Stop") provided in real-time.
- **Visual Feedback:** Color-coded bounding boxes for sighted caregivers or developers.

---

## 📂 Installation & Setup

### For Web/Mobile Version

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abinanthan-CG/streamlit-auto-ocr-live.git
   cd streamlit-auto-ocr-live
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```bash
   python -m streamlit run app.py
   ```

### For Raspberry Pi 5 (Hardware Version)

1. **Prepare your Pi:** Ensure the RPi camera module is connected.
2. **Run the setup script:**
   ```bash
   chmod +x pi_setup.sh
   ./pi_setup.sh
   ```
3. **Run the navigator:**
   ```bash
   python3 pi_navigator.py
   ```

---

## 🧠 Technical Overview

The system uses the **MobileNetSSD** deep learning model to perform real-time object detection.

**The Navigation Logic:**

- **Center Zone Blocked?** System checks Left and Right zones.
- **Left Clear?** Commands: "Move slightly left."
- **Path Clear?** Commands: "Path is clear."
- **Object too large (Close)?** Commands: "STOP! Object in front."

---

## 📦 Requirements

- **Python 3.9+**
- **OpenCV** (DNN Module)
- **Streamlit** (for Web Version)
- **Pyttsx3 / Web Speech API** (depending on version)

---

## 🤝 Contributing

Contributions are welcome! Whether it's adding better distance estimation or new object classes, feel free to fork and PR.

## 📝 License

Distributed under the MIT License.

---

_Developed to assist and empower the visually impaired._
