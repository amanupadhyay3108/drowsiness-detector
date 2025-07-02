import cv2
import winsound
import mediapipe as mp
import streamlit as st
import numpy as np
import math
import time
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Mental Fatigue Detector", layout="centered")
st.title("👁️ Eye Blink Pattern Detector")
st.markdown("Detect mental fatigue based on blink rate and eye closure duration.")

# === Sidebar Settings ===
ear_threshold = st.sidebar.slider("EAR Threshold", min_value=0.1, max_value=0.4, value=0.21, step=0.01)
blink_threshold = st.sidebar.slider("Blinks per Minute Threshold", min_value=1, max_value=30, value=8, step=1)
closure_frame_limit = st.sidebar.slider("Frames for Long Eye Closure", min_value=10, max_value=200, value=90, step=10)
enable_audio = st.sidebar.checkbox("Enable Audio Alert", value=True)

# === Function to calculate EAR ===
def calculate_ear(landmarks, eye_indices, w, h):
    def point(i): return (int(landmarks[i].x * w), int(landmarks[i].y * h))
    p1, p2, p3, p4, p5, p6 = [point(i) for i in eye_indices]
    vertical1 = math.dist(p2, p6)
    vertical2 = math.dist(p3, p5)
    horizontal = math.dist(p1, p4)
    return (vertical1 + vertical2) / (2.0 * horizontal)

# === Function to play alert sound ===
def play_alert():
    winsound.PlaySound("alert.wav", winsound.SND_FILENAME)

# === Constants ===
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
blink_count = 0
start_time = time.time()

closed_eye_frames = 0
open_eye_frames = 0
blink_detected = False

# === MediaPipe setup ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
cap = cv2.VideoCapture(0)

# === Streamlit containers ===
frame_placeholder = st.empty()
status_placeholder = st.empty()
chart_placeholder = st.empty()
table_placeholder = st.empty()
stop = st.button("Stop", key="stop_button_main")

# === Logs ===
log_data = []
blink_history = []

# === App loop ===
while cap.isOpened() and not stop:
    success, frame = cap.read()
    if not success:
        st.error("Failed to access webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    fatigue_status = "Tracking..."
    ear_display = 0.0

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        left_ear = calculate_ear(lm, LEFT_EYE, w, h)
        right_ear = calculate_ear(lm, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        ear_display = avg_ear

        # === Eye open/close tracking ===
        if avg_ear < ear_threshold:
            closed_eye_frames += 1
            open_eye_frames = 0
        else:
            open_eye_frames += 1
            if closed_eye_frames > 2:
                blink_count += 1
            closed_eye_frames = 0

        # === Fatigue evaluation every 60 sec ===
        elapsed_time = time.time() - start_time
        if elapsed_time >= 60:
            fatigue_by_blink = blink_count < blink_threshold
            fatigue_by_closure = closed_eye_frames > closure_frame_limit

            if fatigue_by_blink or fatigue_by_closure:
                fatigue_status = "Fatigued 😴"
                if enable_audio:
                    play_alert()
            else:
                fatigue_status = "Alert 🙂"

            timestamp = datetime.now().strftime("%H:%M:%S")
            log_data.append({
                "Time": timestamp,
                "Blinks/Min": blink_count,
                "Status": fatigue_status
            })
            blink_history.append(blink_count)

            blink_count = 0
            start_time = time.time()

        # === Draw on frame ===
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Fatigue: {fatigue_status}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")
    status_placeholder.info(f"Blink Count: {blink_count} | EAR: {ear_display:.2f}")

    # === Update chart ===
    if blink_history:
        chart_data = pd.DataFrame({"Blinks/Min": blink_history})
        chart_placeholder.line_chart(chart_data)

    # === Update table ===
    if log_data:
        table_placeholder.dataframe(pd.DataFrame(log_data))

cap.release()
cv2.destroyAllWindows()

# === CSV Export ===
if log_data:
    st.markdown("### 📥 Download Fatigue Log")
    df_log = pd.DataFrame(log_data)
    csv = df_log.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="fatigue_log.csv", mime="text/csv")