import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import os

# 1. Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 2. Piano Key Configuration (Name, x_start, x_end, Sound_File)
PIANO_KEYS = [
    ["C", 50, 150, "C.wav"],
    ["D", 160, 260, "D.wav"],
    ["E", 270, 370, "E.wav"],
    ["F", 380, 480, "F.wav"],
    ["G", 490, 590, "G.wav"]
]

def process_frame(image):
    if image is None:
        return None, None
    
    # Mirror the image for intuitive movement
    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    active_note = None

    # Draw the Piano Interface (Static Keys)
    for key in PIANO_KEYS:
        name, x1, x2, sound = key
        cv2.rectangle(image, (x1, 50), (x2, 250), (255, 255, 255), 2)
        cv2.putText(image, name, (x1 + 35, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Hand Tracking and Collision Logic
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Index finger tip is landmark 8
            index_tip = hand_lms.landmark[8]
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            
            # Draw green pointer on fingertip
            cv2.circle(image, (ix, iy), 15, (0, 255, 0), -1)

            # Check if fingertip is inside any key rectangle
            for key in PIANO_KEYS:
                name, x1, x2, sound = key
                if x1 < ix < x2 and 50 < iy < 250:
                    # Highlight key Green and trigger sound
                    cv2.rectangle(image, (x1, 50), (x2, 250), (0, 255, 0), -1)
                    cv2.putText(image, name, (x1 + 35, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    
                    # Verify file exists before trying to play it
                    if os.path.exists(sound):
                        active_note = sound 

    return image, active_note

# 3. Gradio Interface Setup (Only ONE interface definition needed)
interface = gr.Interface(
    fn=process_frame,
    inputs=gr.Image(sources="webcam", streaming=True),
    outputs=[
        gr.Image(label="Air Piano Display"),
        gr.Audio(label="Sound Output", autoplay=True)
    ],
    live=True,
    title="Python Air Hand Piano",
    description="Use your index finger to play the piano! Ensure C.wav through G.wav are in your folder."
)

if __name__ == "__main__":
    interface.launch()
