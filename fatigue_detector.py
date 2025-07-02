import cv2
import mediapipe as mp
import math
import time

def calculate_ear(landmarks, eye_indices, w, h):
    def point(index):
        return (int(landmarks[index].x * w), int(landmarks[index].y * h))

    p1, p2, p3, p4, p5, p6 = [point(i) for i in eye_indices]
    vertical1 = math.dist(p2, p6)
    vertical2 = math.dist(p3, p5)
    horizontal = math.dist(p1, p4)
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# Eye landmark indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
ear_threshold = 0.21

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
cap = cv2.VideoCapture(0)

# Blink + Time Tracking
blink_count = 0
blink_detected = False
start_time = time.time()
blink_history = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        left_ear = calculate_ear(lm, LEFT_EYE, w, h)
        right_ear = calculate_ear(lm, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2

        if avg_ear < ear_threshold:
            if not blink_detected:
                blink_count += 1
                blink_detected = True
        else:
            blink_detected = False

        # Record blink per minute
        elapsed_time = time.time() - start_time
        if elapsed_time >= 60:
            blink_history.append(blink_count)
            print(f"Blinks this minute: {blink_count}")
            if blink_count < 8:
                fatigue_status = "Fatigued 😴"
            else:
                fatigue_status = "Alert 🙂"

            # Reset for next minute
            start_time = time.time()
            blink_count = 0
        else:
            fatigue_status = "Tracking..."

        # Display status
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Fatigue: {fatigue_status}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Fatigue Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
