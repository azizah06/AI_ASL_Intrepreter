import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ====== Parameter =======
img_size = 96  # MobileNetV2 minimal input size 96x96
batch_size = 32
epochs = 25
dataset_dir = "./dataset_bisindo"  # sesuaikan path dataset kamu

# ====== Data Augmentation =======
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ====== Load base model MobileNetV2 tanpa top layer =======
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_size, img_size, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base_model weights supaya tidak langsung berubah saat awal training
base_model.trainable = False

# ====== Buat model baru =======
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# ====== Training tahap 1: hanya train dense layers =======
print("🚀 Training tahap 1: hanya dense layers (fine-tune tahap awal)...")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[early_stop, reduce_lr]
)

# ====== Training tahap 2: buka beberapa layer MobileNet dan fine-tune =======
print("🚀 Training tahap 2: fine-tune beberapa layer terakhir MobileNetV2...")

base_model.trainable = True

# Freeze semua layer kecuali 50 layer terakhir untuk fine-tune
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # pakai learning rate kecil untuk fine-tune
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs - 10,
    callbacks=[early_stop, reduce_lr]
)

# ====== Plot hasil training =======
plt.plot(history1.history['accuracy'] + history2.history['accuracy'], label='Training Accuracy')
plt.plot(history1.history['val_accuracy'] + history2.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ====== Simpan model =======
model.save("bisindo_mobilenetv2_finetune.h5")
print("💾 Model saved as 'bisindo_mobilenetv2_finetune.h5'")
