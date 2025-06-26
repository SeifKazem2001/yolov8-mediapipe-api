from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from ultralytics import YOLO
import mediapipe as mp
from typing import Optional

# Load YOLO model
model = YOLO('best4.pt')

# Load MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Arabic object labels
object_names_arabic = {
    "person": "شخص",
    "car": "سيارة", 
    "bus": "حافلة",
    "truck": "شاحنة",
    "bike": "دراجة",
    "phone": "هاتف"
    # Add more if needed
}

app = FastAPI()

# Count fingers utility
def count_fingers(hand_landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [2, 6, 10, 14, 18]
    count = 0
    if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_pips[0]].x:
        count += 1
    for i in range(1, 5):
        if hand_landmarks.landmark[finger_tips[i]].y < hand_landmarks.landmark[finger_pips[i]].y:
            count += 1
    return count

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running 🚀"}



@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    # Read image into numpy
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model.predict(img, imgsz=640, conf=0.3)
    detected_object = None
    if results and results[0].boxes:
        cls_id = int(results[0].boxes[0].cls[0])
        detected_object = model.names[cls_id]
        object_label_ar = object_names_arabic.get(detected_object, detected_object)
    else:
        object_label_ar = "لا يوجد كائن"

    # Hand detection
    """img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(img_rgb)
    finger_count = 0
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            finger_count = count_fingers(hand_landmarks)
            break"""
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(img_rgb)
    finger_count = 0
    hand_label = "لا يوجد"

    hand_label_arabic = {"Left": "يسار", "Right": "يمين"}

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            finger_count = count_fingers(hand_landmarks)
            hand_en = handedness.classification[0].label  # "Left" or "Right"
            hand_label = hand_label_arabic.get(hand_en, hand_en)
            break  # نكتفي بأول يد


    return {
        "object": object_label_ar,
        "fingers": finger_count,
        "hand":hand_label
    }
@app.post("/level2/")
async def count_fingers_only(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(img_rgb)

    finger_count = 0
    hand_label = "لا يوجد"
    hand_label_arabic = {"Left": "يسار", "Right": "يمين"}

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            finger_count = count_fingers(hand_landmarks)
            hand_en = handedness.classification[0].label
            hand_label = hand_label_arabic.get(hand_en, hand_en)
            break  # نأخذ أول يد فقط

    return {
        "fingers": finger_count,
        "hand": hand_label
    }

#api level 1  object and right left 
#api level 2 finglers