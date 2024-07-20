import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image, ExifTags
from io import BytesIO
import requests
import mediapipe as mp
import gdown
import os

st.set_page_config(page_title="SIBI Classification App", page_icon="🤟")

# Daftar URL gambar untuk setiap kelas
image_urls = {
    "A": "https://drive.google.com/uc?export=view&id=1WnCRi8Gle_SKR4vr2GkHEZ74wNIefkkw",
    "B": "https://drive.google.com/uc?export=view&id=1QXj4oh1IOIoctQmnsgzHD63IgBsrt8Qd",
    "C": "https://drive.google.com/uc?export=view&id=1a1LDkRN-VXHd4MdwFNcRibW1Se7DI3B_",
    "D": "https://drive.google.com/uc?export=view&id=1-IxCobV_8pmbci3GjrTkWw5MipBwLslK",
    "E": "https://drive.google.com/uc?export=view&id=12_bM-gYdW-nYP97fqMqr4FA-ZiDnMgGL",
    "F": "https://drive.google.com/uc?export=view&id=1SOnxIRnwwkvmnvJyUzYPCtFrnKBlxVv7",
    "G": "https://drive.google.com/uc?export=view&id=1Qwm-8iICfjczv-tXjyWaieB1O_WX5xod",
    "H": "https://drive.google.com/uc?export=view&id=1tq8oGvH7qN7UkJdK71b6xYGnzpE4n0L4",
    "I": "https://drive.google.com/uc?export=view&id=1RGZaz_ByzqVnGqcPP6QPxgYdjSIZgtok",
    "K": "https://drive.google.com/uc?export=view&id=1BW94bUHP0Mxjc2MN6Ft2fXBDYeYRpdJN",
    "L": "https://drive.google.com/uc?export=view&id=18MlNmoXVIbO8hFJzofLa6-EGXNhzJu3c",
    "M": "https://drive.google.com/uc?export=view&id=1vgIokxNVf3HmXqJfwswgdo4A7qVCMo3E",
    "N": "https://drive.google.com/uc?export=view&id=1hV1bwseyKvs3oPZWEg2Qbde9ryQ12q3p",
    "O": "https://drive.google.com/uc?export=view&id=1x7Cb4RN2_z_K1yufa2GiFv9qX5NKRAYz",
    "P": "https://drive.google.com/uc?export=view&id=1qdgA7dhgnCnRFRlhCRKcfPcgT65wB8gR",
    "Q": "https://drive.google.com/uc?export=view&id=1MA1vxhOzqarFQYJC6xwxXK4cqanrBkaX",
    "R": "https://drive.google.com/uc?export=view&id=1wArg9ptsvTbOc8l4LFXE_ca7o67c_YNb",
    "S": "https://drive.google.com/uc?export=view&id=1gP52H7MLkfcb6Y3dOKmj3sgr5Sjnptam",
    "T": "https://drive.google.com/uc?export=view&id=1gdo0IyWSYj0cOw8V7Mv-bmIckQG6_fHv",
    "U": "https://drive.google.com/uc?export=view&id=1RwEGVSCxPLuK4oSJIGp6I1YZraq5zKKI",
    "V": "https://drive.google.com/uc?export=view&id=1fNwZVnzW8s0uvG8MY1rdl80v6P9Wp4j3",
    "W": "https://drive.google.com/uc?export=view&id=1N7IwI3KXL15ZeeGUDkQ3cdd1nQPe-R2b",
    "X": "https://drive.google.com/uc?export=view&id=1YVpIs4ZxXQs0O1suzJa1bhMaYPSfmkAv",
    "Y": "https://drive.google.com/uc?export=view&id=1VXZrNEnDAC1bDQQ8hJPrAayQLfGJS8uJ"
}

# Kamus untuk memetakan indeks ke huruf
index_to_label = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 
                  8: "I", 9: "K", 10: "L", 11: "M", 12: "N", 13: "O", 14: "P", 
                  15: "Q", 16: "R", 17: "S", 18: "T", 19: "U", 20: "V", 
                  21: "W", 22: "X", 23: "Y"}

# About the dataset
about = """
Bahasa Isyarat merupakan bahasa yang diproduksi menggunakan gerakan
tangan (gestur) dan dipersepsi menggunakan indra penglihatan untuk saling
mengidentifikasi dan memperoleh informasi. Bahasa ini banyak digunakan oleh
penyandang disabilitas tuli atau tuna rungu untuk berkomunikasi.

Sistem Isyarat Bahasa Indonesia (SIBI) adalah bahasa formal yang
diresmikan oleh Kementerian Pendidikan dan Kebudayaan pada tahun 1997 yang
diadopsi dari American Sign Language atau dikenal dengan sebutan ASL. SIBI
merupakan bahasa isyarat dengan menggunakan tangan kanan untuk menunjukkan
tulisan alfabet.
"""

# Fungsi untuk memperbaiki orientasi gambar
def correct_image_orientation(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(img._getexif().items())
        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return img

# Fungsi untuk menampilkan halaman landing page
def landing_page():
    st.title("Sistem Isyarat Bahasa Indonesia (SIBI)")
    st.header("Perkenalan Dataset")
    st.markdown(about)
    st.write("Berikut adalah contoh gestur tangan untuk setiap huruf dalam alfabet Bahasa Isyarat Indonesia (SIBI):")

    cols = st.columns(4)
    for idx, (label, url) in enumerate(image_urls.items()):
        col = cols[idx % 4]
        with col:
            st.write(f"Mewakili huruf :  {label}")
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img = correct_image_orientation(img)
            st.image(img, use_column_width=True)


# Fungsi untuk mengunduh model dari Google Drive
def download_from_drive(url, output_path):
    if not os.path.exists(output_path):
        st.write(f"Mengunduh model dari {url}...")
        gdown.download(url, output_path, quiet=False)
    if os.path.exists(output_path):
        st.write(f"Model berhasil diunduh: {output_path}")
    else:
        st.write(f"Model tidak ditemukan: {output_path}")
    return output_path

# Fungsi untuk memuat model menggunakan cache
@st.cache_resource
def load_model_from_drive(url, output_path):
    model_path = download_from_drive(url, output_path)
    st.write(f"Muat model dari {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        st.write("Model berhasil dimuat.")
        return model
    except Exception as e:
        st.write(f"Error memuat model: {e}")
        return None

# URL Google Drive dan jalur file lokal untuk model
vgg16_url = 'https://drive.google.com/uc?id=16DHxT0lAEwjK1ok7Fso5uqp1P-tT8VXi'
vgg16_path = 'VGG16.keras'  # Tidak menggunakan sub-folder
vgg19_url = 'https://drive.google.com/uc?id=1hAMEoSQvu2IcGBVAggokrIOGzqK9qb47'
vgg19_path = 'VGG19.keras'  # Tidak menggunakan sub-folder

# Memuat model dengan cache
model1 = load_model_from_drive(vgg16_url, vgg16_path)
model2 = load_model_from_drive(vgg19_url, vgg19_path)
models = {"VGG16": model1, "VGG19": model2}

if model1 is None or model2 is None:
    st.write("Ada masalah dalam memuat model.")
else:
    st.write("Semua model berhasil dimuat.")

def webcam_classification_page(models):
    st.title("Webcam Classification")
    st.write("Use your webcam to classify images using a chosen model.")

    model_choice = st.selectbox("Choose a model", models.keys())
    model = models[model_choice]

    run = st.button("Start Webcam")
    FRAME_WINDOW = st.image([])

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                cx_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                cy_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                cx_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                cy_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                hand_img = frame_rgb[cy_min:cy_max, cx_min:cx_max]

                if hand_img.size != 0:
                    hand_img = cv2.resize(hand_img, (128, 128))
                    hand_img = np.expand_dims(hand_img, axis=0)
                    hand_img = hand_img / 255.0

                    prediction = model.predict(hand_img)
                    predicted_index = np.argmax(prediction)
                    predicted_label = index_to_label[predicted_index]

                    cv2.putText(frame_rgb, predicted_label, (cx_min, cy_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        FRAME_WINDOW.image(frame_rgb)

    camera.release()

def upload_classification_page(models):
    st.title("Image Upload Classification")
    st.write("Upload an image to classify it using both models.")

    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        img_array = np.array(image.resize((128, 128)))/255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions_model1 = models["VGG16"].predict(img_array)
        predictions_model2 = models["VGG19"].predict(img_array)
        
        label_model1 = np.argmax(predictions_model1)
        label_model2 = np.argmax(predictions_model2)
        
        confidence_model1 = np.max(predictions_model1) * 100
        confidence_model2 = np.max(predictions_model2) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Model VGG16: {index_to_label[label_model1]}")
            st.write(f"Confidence: {confidence_model1:.2f}%")
        
        with col2:
            st.write(f"Model VGG19: {index_to_label[label_model2]}")
            st.write(f"Confidence: {confidence_model2:.2f}%")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Landing Page", "Webcam Classification", "Image Upload Classification"])

    if page == "Landing Page":
        landing_page()
    elif page == "Webcam Classification":
        webcam_classification_page(models)
    elif page == "Image Upload Classification":
        upload_classification_page(models)

if __name__ == "__main__":
    main()