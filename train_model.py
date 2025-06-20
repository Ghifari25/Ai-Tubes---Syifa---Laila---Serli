# train_model.py
import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Path ke direktori training
train_dir = "Test"
img_height, img_width = 100, 100
batch_size = 16
epochs = 10
learning_rate = 0.0005

# Image Preprocessing dan augmentasi
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Simpan class indices ke file JSON
output_dir = "nn_streamlit_app"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "class_indices.json"), "w") as f:
    json.dump(train_generator.class_indices, f)

# Bangun model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training model
model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# Simpan model
model.save(os.path.join(output_dir, "best_model.keras"))

# Evaluasi model
val_loss, val_accuracy = model.evaluate(val_generator)

# Simpan akurasi validasi
with open(os.path.join(output_dir, "accuracy.json"), "w") as f:
    json.dump({"validation_accuracy": val_accuracy}, f)
