import cv2
import numpy as np
import pyttsx3
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import json
import time
import threading
import mediapipe as mp

# === Load model dan label BISINDO ===
model = tf.keras.models.load_model("bisindo_mobilenetv2_finetune.h5")  # Ganti ke model BISINDO

with open("class_indices.json", "r") as f:
    label_map = json.load(f)

# Balik mapping: index -> label
class_indices = {v: k for k, v in label_map.items()}

# === Text-to-Speech ===
engine = pyttsx3.init()
last_pred = ''
last_time = 0
delay = 1.5  # delay antar prediksi suara

def speak(text):
    engine.say(text)
    engine.runAndWait()

# === Mediapipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Ubah max_num_hands jadi 2
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# === Mulai Webcam ===
cap = cv2.VideoCapture(0)
print("🎥 Webcam BISINDO dimulai. Tekan 'q' atau tutup jendela untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Gagal membaca frame.")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    roi = None
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        # Kumpulkan semua koordinat x dan y dari semua tangan
        x_coords_all = []
        y_coords_all = []

        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            x_coords_all.extend(x_coords)
            y_coords_all.extend(y_coords)

            # Gambar landmark tangan (opsional)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Bounding box gabungan untuk semua tangan
        x_min, x_max = int(min(x_coords_all)), int(max(x_coords_all))
        y_min, y_max = int(min(y_coords_all)), int(max(y_coords_all))

        margin = 40
        x1 = max(x_min - margin, 0)
        y1 = max(y_min - margin, 0)
        x2 = min(x_max + margin, w)
        y2 = min(y_max + margin, h)

        # Gambar bounding box gabungan
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        roi = frame[y1:y2, x1:x2]

    if roi is not None:
        try:
            image = cv2.resize(roi, (96, 96))
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
            print("❌ Gagal memproses ROI:", e)

    cv2.imshow("BISINDO Detection", frame)

    # Keluar jika tekan 'q' atau jendela ditutup
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Keluar (tekan 'q').")
        break
    if cv2.getWindowProperty("BISINDO Detection", cv2.WND_PROP_VISIBLE) < 1:
        print("❌ Jendela ditutup.")
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
