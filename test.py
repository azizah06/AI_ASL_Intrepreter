import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ==== Load model dan label ====
model = load_model("asl_cnn_model.h5")
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# ==== Load dan preprocess gambar ====
img_path = "./dataset/asl_alphabet_test/A4_test.jpg"  # Ganti dengan path ke gambar uji
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img) / 255.0  # Normalisasi
img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch

# ==== Prediksi ====
pred = model.predict(img_array)
predicted_class = class_names[np.argmax(pred)]
confidence = np.max(pred)

print(f"✅ Prediksi huruf: {predicted_class} ({confidence*100:.1f}%)")
