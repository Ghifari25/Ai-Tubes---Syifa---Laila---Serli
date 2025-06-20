import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
from flask import Flask, request, render_template

# Judul
st.title("🧠 Deteksi Disleksia dari Tulisan")

# Versi TensorFlow
st.write("📦 TensorFlow Version:", tf.__version__)

# Path model, label, dan akurasi
model_path = os.path.join("nn_streamlit_app", "best_model.keras")
class_indices_path = os.path.join("nn_streamlit_app", "class_indices.json")
accuracy_path = os.path.join("nn_streamlit_app", "accuracy.json")

# Tampilkan akurasi jika tersedia
if os.path.exists(accuracy_path):
    with open(accuracy_path) as f:
        acc_data = json.load(f)
        acc_val = acc_data.get("validation_accuracy", None)
        if acc_val is not None:
            st.info(f"📈 Akurasi Validasi Model: **{acc_val * 100:.2f}%**")

if os.path.exists(model_path) and os.path.exists(class_indices_path):
    model = tf.keras.models.load_model(model_path)
    with open(class_indices_path) as f:
        class_indices = json.load(f)

    # Urutkan nama kelas berdasarkan indeks
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

    # Upload gambar
    uploaded_file = st.file_uploader("📤 Upload gambar tulisan (.jpg/.png)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="📷 Gambar yang Diupload", use_container_width=True)

        # Preprocessing
        img_resized = image.resize((100, 100))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        prediction = model.predict(img_array)[0]
        predicted_label = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.success(f"🧾 Hasil: **{predicted_label}**")
        st.write(f"📊 Tingkat Evaluasi Akurasi: **{confidence * 100:.2f}%**")

        # Probabilitas semua kelas
        st.subheader("🔍 Probabilitas Setiap Kelas:")
        for i, prob in enumerate(prediction):
            st.write(f"- **{class_names[i]}**: {prob * 100:.2f}%")
else:
    st.error("❌ Model atau label tidak ditemukan.")
    st.info("Silakan jalankan `train_model.py` untuk melatih model terlebih dahulu.")
