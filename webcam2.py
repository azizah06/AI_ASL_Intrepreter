import cv2
import numpy as np
import pyttsx3
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import json
import time
import threading
import mediapipe as mp

# === Load model dan label ===
model = tf.keras.models.load_model("asl_cnn_model.h5")

# Load class indices dari file JSON (mapping label ke index)
with open("class_indices.json", "r") as f:
    label_map = json.load(f)

# Balik mapping agar index -> label, misal {0: 'A', 1: 'B', ...}
class_indices = {v: k for k, v in label_map.items()}

# === Text-to-Speech Engine ===
engine = pyttsx3.init()
last_pred = ''
last_time = 0
delay = 1.5  # jeda antar suara (detik)

# === Mediapipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === Fungsi untuk suara (threading supaya tidak lag) ===
def speak(text):
    engine.say(text)
    engine.runAndWait()

# === Mulai Webcam ===
cap = cv2.VideoCapture(0)
print("🎥 Webcam dimulai. Tekan 'q' atau tutup jendela untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Gagal membaca frame dari webcam.")
        break

    # Flip horizontal untuk tampilan mirror
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    roi = None
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            margin = 40
            x1 = max(x_min - margin, 0)
            y1 = max(y_min - margin, 0)
            x2 = min(x_max + margin, w)
            y2 = min(y_max + margin, h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            roi = frame[y1:y2, x1:x2]
            break

    if roi is not None:
        try:
            image = cv2.resize(roi, (64, 64))
            image = img_to_array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            preds = model.predict(image, verbose=0)
            pred_idx = np.argmax(preds)
            pred_label = class_indices[pred_idx]
            confidence = preds[0][pred_idx]

            current_time = time.time() 
            if pred_label != last_pred and (current_time - last_time) > delay:
                threading.Thread(target=speak, args=(pred_label,), daemon=True).start()
                last_pred = pred_label
                last_time = current_time

            cv2.putText(frame, f"Predicted: {pred_label} ({confidence:.2f})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except Exception as e:
            print("❌ Gagal proses ROI:", e)

    cv2.imshow("ASL Detection", frame)

    # Keluar jika tekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Keluar dari webcam (tekan q).")
        break

    # Keluar jika jendela ditutup secara manual
    if cv2.getWindowProperty("ASL Detection", cv2.WND_PROP_VISIBLE) < 1:
        print("❌ Jendela ditutup secara manual.")
        break

# === Bersihkan ===
cap.release()
cv2.destroyAllWindows()
