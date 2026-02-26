#!/bin/bash

echo "--- Installing Dependencies for SEEING WITH SOUND (RPi 5) ---"

# Update system
sudo apt-get update
sudo apt-get install -y libcamera-apps espeak-ng python3-opencv

# Install Python requirements
pip3 install pyttsx3 numpy picamera2

echo "--- Setup Complete ---"
echo "To run the navigator: python3 pi_navigator.py"
echo "To see the preview (on desktop): python3 pi_navigator.py --display"
