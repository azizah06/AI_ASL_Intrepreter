import cv2

cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("✅ Webcam tersedia dan bisa digunakan.")
else:
    print("❌ Webcam tidak tersedia atau tidak terdeteksi.")

cap.release()
