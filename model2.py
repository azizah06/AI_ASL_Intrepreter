import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ==== PARAMETER ====
img_size = 64
batch_size = 32
epochs = 10
dataset_dir = "./dataset/asl_alphabet_train"

# ==== PREPROCESSING ====
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
# ✅ Simpan mapping class ke file
with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)
print("📁 class_indices.json saved.")

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False  
)

# ==== MODEL CNN ====
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==== EARLY STOPPING ====
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# ==== TRAINING ====
print("🚀 Training model...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[early_stop]
)

# ==== EVALUASI ====
val_loss, val_accuracy = model.evaluate(val_gen)
print(f"\n✅ Validation Accuracy: {val_accuracy:.2f}")
print(f"📉 Validation Loss: {val_loss:.2f}")

# ==== GRAFIK AKURASI DAN LOSS ====
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_evaluation.png')  # Simpan grafik ke file
plt.show()
print("💾 Grafik training dan validation disimpan sebagai 'training_evaluation.png'")

# ==== CONFUSION MATRIX DAN METRIK ====
# Reset generator supaya mulai dari awal (pastikan shuffle=False saat membuat val_gen)
val_gen.reset()
Y_true = val_gen.classes
Y_pred_prob = model.predict(val_gen, verbose=1)
Y_pred = np.argmax(Y_pred_prob, axis=1)

# Mapping class indices ke nama label
class_labels = list(train_gen.class_indices.keys())

# Confusion matrix
cm = confusion_matrix(Y_true, Y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Simpan confusion matrix ke file
plt.show()
print("💾 Confusion matrix disimpan sebagai 'confusion_matrix.png'")

# Classification report (termasuk recall dan f1-score)
report = classification_report(Y_true, Y_pred, target_names=class_labels)
print("📊 Classification Report:\n", report)

# Simpan classification report ke file txt
with open("classification_report.txt", "w") as f:
    f.write(report)
print("💾 Classification report disimpan sebagai 'classification_report.txt'")

# ==== SIMPAN MODEL ====
model.save("asl_cnn_model.h5")
print("💾 Model saved as 'asl_cnn_model.h5'")
