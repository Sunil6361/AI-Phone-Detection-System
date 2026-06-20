import os
import time
import cv2
import pygame
from ultralytics import YOLO

# =========================
# CONFIGURATION
# =========================
MODEL_FILE = "yolov8s.pt"   # better accuracy than yolov8n
ALERT_SOUND = "alert.wav"
CONFIDENCE_LEVEL = 0.55
COOLDOWN_SECONDS = 2
FRAME_SKIP = 1   # process every frame for accuracy

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_FILE)

# =========================
# AUDIO SETUP
# =========================
pygame.mixer.init()
sound_loaded = False

if os.path.exists(ALERT_SOUND):
    try:
        pygame.mixer.music.load(ALERT_SOUND)
        sound_loaded = True
    except Exception as e:
        print("Audio load error:", e)

# =========================
# CAMERA SETUP
# =========================
cap = cv2.VideoCapture(1)

# Set resolution (improves detection)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Camera not found")
    exit()

print("Press 'q' to quit")

last_alert_time = 0

# =========================
# FUNCTION: CHECK NEAR
# =========================
def is_phone_near_person(person, phone):
    px1, py1, px2, py2 = person
    fx1, fy1, fx2, fy2 = phone

    cx = (fx1 + fx2) // 2
    cy = (fy1 + fy2) // 2

    margin = 50  # expand detection area

    return (
        px1 - margin <= cx <= px2 + margin and
        py1 - margin <= cy <= py2 + margin
    )

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    persons = []
    phones = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = model.names[cls_id]

            if conf < CONFIDENCE_LEVEL:
                continue

            if label == "person":
                persons.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            elif label == "cell phone":
                phones.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Check proximity
    for person in persons:
        for phone in phones:
            if is_phone_near_person(person, phone):

                cv2.putText(frame, "PHONE NEAR PERSON!",
                            (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255), 3)

                current_time = time.time()

                if current_time - last_alert_time > COOLDOWN_SECONDS:
                    print(">>> ALERT!")

                    if sound_loaded:
                        pygame.mixer.music.play()

                    last_alert_time = current_time

    cv2.imshow("No Phone Zone", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()