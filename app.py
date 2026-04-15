import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import av
import pygame
import time

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Air Piano", layout="wide")
st.title("AI Air Piano")
st.write("Use your **Index Finger** to press virtual keys")

# ---------------- LOAD SOUND ----------------
pygame.mixer.init()

sounds = {
    "C": pygame.mixer.Sound("C.wav"),
    "D": pygame.mixer.Sound("D.wav"),
    "E": pygame.mixer.Sound("E.wav"),
    "F": pygame.mixer.Sound("F.wav"),
}

# ---------------- VIDEO PROCESSOR ----------------
class PianoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.last_note = None
        self.last_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        h, w, _ = img.shape

        # -------- RESPONSIVE KEYS --------
        key_width = w // 6
        keys = {
            "C": [key_width*0, int(h*0.3), key_width*1, int(h*0.8)],
            "D": [key_width*1, int(h*0.3), key_width*2, int(h*0.8)],
            "E": [key_width*2, int(h*0.3), key_width*3, int(h*0.8)],
            "F": [key_width*3, int(h*0.3), key_width*4, int(h*0.8)],
        }

        # -------- HAND TRACKING --------
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        current_note = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                itip = hand_landmarks.landmark[8]
                cx, cy = int(itip.x * w), int(itip.y * h)

                # Draw finger
                cv2.circle(img, (cx, cy), 12, (0, 255, 0), cv2.FILLED)

                # -------- COLLISION --------
                for note, rect in keys.items():
                    if rect[0] < cx < rect[2] and rect[1] < cy < rect[3]:
                        current_note = note

        # -------- DEBOUNCE + SOUND --------
        current_time = time.time()

        if current_note != self.last_note:
            if current_note and (current_time - self.last_time) > 0.3:
                sounds[current_note].play()
                self.last_time = current_time

            self.last_note = current_note

        # -------- DRAW KEYS --------
        for note, rect in keys.items():
            color = (255, 255, 255)

            if note == current_note:
                color = (0, 255, 0)

            # Filled
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, cv2.FILLED)
            # Border
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0), 2)

            # Label
            cv2.putText(img, note, (rect[0]+20, rect[3]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # -------- DISPLAY NOTE --------
        if current_note:
            cv2.putText(img, f"Playing: {current_note}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def __del__(self):
        self.hands.close()

# ---------------- START STREAM ----------------
webrtc_streamer(
    key="air-piano",
    video_processor_factory=PianoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
