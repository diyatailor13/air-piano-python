import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import av

# --- INITIALIZE MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

st.set_page_config(page_title="AI Air Piano", layout="wide")
st.title("Python Air Piano")
st.write("Position your hand so your **Index Finger** hits the boxes!")

class PianoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror effect
        h, w, _ = img.shape

        # Define 4 Keys [x1, y1, x2, y2]
        keys = {
            "C": [50, 100, 150, 300],
            "D": [160, 100, 260, 300],
            "E": [270, 100, 370, 300],
            "F": [380, 100, 480, 300]
        }

        # Draw Piano Interface
        for note, rect in keys.items():
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 255), 2)
            cv2.putText(img, note, (rect[0]+30, rect[3]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Hand Tracking Logic
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Landmark 8 is the Index Finger Tip
                itip = hand_landmarks.landmark[8]
                cx, cy = int(itip.x * w), int(itip.y * h)
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

                # Collision Detection
                for note, rect in keys.items():
                    if rect[0] < cx < rect[2] and rect[1] < cy < rect[3]:
                        # Highlight key when pressed
                        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), cv2.FILLED)
                        # Note: Server-side audio requires WebRTC data channel for low latency

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="air-piano", video_processor_factory=PianoProcessor)
