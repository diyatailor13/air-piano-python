import gradio as gr
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Piano Key Configuration
# Format: [Name, x_start, x_end, Sound_Path]
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
    
    # Flip the image for mirror effect
    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    active_note = None

    # 1. Draw the Piano Interface
    for key in PIANO_KEYS:
        name, x1, x2, sound = key
        # Draw white key rectangles
        cv2.rectangle(image, (x1, 50), (x2, 250), (255, 255, 255), 2)
        cv2.putText(image, name, (x1 + 35, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 2. Hand Tracking Logic
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Index finger tip is landmark 8
            index_tip = hand_lms.landmark[8]
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            
            # Draw pointer on finger
            cv2.circle(image, (ix, iy), 15, (0, 255, 0), -1)

            # Check for collision with keys
            for key in PIANO_KEYS:
                name, x1, x2, sound = key
                if x1 < ix < x2 and 50 < iy < 250:
                    # Highlight the key in Green
                    cv2.rectangle(image, (x1, 50), (x2, 250), (0, 255, 0), -1)
                    cv2.putText(image, name, (x1 + 35, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    active_note = sound # This triggers the audio output

    return image, active_note

# Gradio Interface Setup
interface = gr.Interface(
    fn=process_frame,
    inputs=gr.Image(sources="webcam", streaming=True),
    outputs=[
        gr.Image(label="Air Piano Display"),
        gr.Audio(label="Sound Output", autoplay=True)
    ],
    live=True,
    title="Python Air Hand Piano",
    description="Hold your index finger up and 'touch' the virtual keys to play!"
)

if __name__ == "__main__":
    interface.launch()
