import cv2
import mediapipe as mp
import math

# ---------- EAR Calculation Function ----------
def calculate_ear(landmarks, eye_indices, image_w, image_h):
    def get_point(index):
        return (int(landmarks[index].x * image_w), int(landmarks[index].y * image_h))

    p1 = get_point(eye_indices[0])
    p2 = get_point(eye_indices[1])
    p3 = get_point(eye_indices[2])
    p4 = get_point(eye_indices[3])
    p5 = get_point(eye_indices[4])
    p6 = get_point(eye_indices[5])

    # Calculate Euclidean distances
    vertical_1 = math.dist(p2, p6)
    vertical_2 = math.dist(p3, p5)
    horizontal = math.dist(p1, p4)

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# ---------- Setup ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
cap = cv2.VideoCapture(0)

# Eye landmark indices (MediaPipe)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

blink_count = 0
ear_threshold = 0.21  # Adjust if needed
blink_detected = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Calculate EAR for both eyes
        right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
        left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
        avg_ear = (right_ear + left_ear) / 2.0

        # Blink Detection Logic
        if avg_ear < ear_threshold:
            if not blink_detected:
                blink_count += 1
                blink_detected = True
        else:
            blink_detected = False

        # Display blink count and EAR
        cv2.putText(frame, f"Blinks: {blink_count}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Eye Blink Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
