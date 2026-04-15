import gradio as gr
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def air_piano(image):
    # image is a numpy array from the webcam
    h, w, c = image.shape
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Define Piano Keys (Visual only for now)
    # On a web server, playing sounds instantly is tricky via OpenCV
    # We draw the keys on the image to show they are being "hit"
    cv2.rectangle(image, (50, 50), (150, 200), (255, 255, 255), 2) # Key C
    cv2.putText(image, "C", (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            index_tip = hand_lms.landmark[8]
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(image, (ix, iy), 10, (0, 255, 0), -1)
            
            # Simple Collision Logic
            if 50 < ix < 150 and 50 < iy < 200:
                cv2.rectangle(image, (50, 50), (150, 200), (0, 255, 0), -1)
                # Note: In a real web app, you trigger a JS sound here
                
    return image

# Gradio Interface
interface = gr.Interface(
    fn=air_piano, 
    inputs=gr.Image(sources="webcam", streaming=True), 
    outputs="image",
    live=True
)

interface.launch()
