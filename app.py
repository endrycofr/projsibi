import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image, ExifTags
import mediapipe as mp
import os
import time

st.set_page_config(page_title="SIBI Classification App", page_icon="🤟")

# Daftar path gambar untuk setiap kelas
image_paths = {
    "A": "sample/A.jpg",
    "B": "sample/B.jpg",
    "C": "sample/C.jpg",
    "D": "sample/D.jpg",
    "E": "sample/E.jpg",
    "F": "sample/F.jpg",
    "G": "sample/G.jpg",
    "H": "sample/H.jpg",
    "I": "sample/I.jpg",
    "K": "sample/K.jpg",
    "L": "sample/L.jpg",
    "M": "sample/M.jpg",
    "N": "sample/N.jpg",
    "O": "sample/O.jpg",
    "P": "sample/P.jpg",
    "Q": "sample/Q.jpg",
    "R": "sample/R.jpg",
    "S": "sample/S.jpg",
    "T": "sample/T.jpg",
    "U": "sample/U.jpg",
    "V": "sample/V.jpg",
    "W": "sample/W.jpg",
    "X": "sample/X.jpg",
    "Y": "sample/Y.jpg"
}

# Kamus untuk memetakan indeks ke huruf
index_to_label = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 
                  8: "I", 9: "K", 10: "L", 11: "M", 12: "N", 13: "O", 14: "P", 
                  15: "Q", 16: "R", 17: "S", 18: "T", 19: "U", 20: "V", 
                  21: "W", 22: "X", 23: "Y"}

# About the dataset
about = """
Pengenalan Bahasa Isyarat SIBI
Selamat datang di platform kami yang bertujuan untuk memudahkan komunikasi dengan penyandang disabilitas melalui pengenalan bahasa isyarat SIBI (Sistem Isyarat Bahasa Indonesia). Bahasa isyarat adalah bentuk komunikasi yang menggunakan gerakan tangan untuk menyampaikan pesan, terutama digunakan oleh teman-teman kita yang memiliki hambatan pendengaran.

Cara Kerja
Website kami dirancang untuk membantu Anda mengenali huruf-huruf dalam bahasa isyarat SIBI dengan mudah dan cepat. Terdapat dua metode yang bisa Anda gunakan:

Penggunaan Kamera Real-Time: Cukup aktifkan kamera Anda, dan sistem kami akan langsung mengenali gerakan tangan Anda. Dalam waktu singkat, huruf yang Anda isyaratkan akan muncul di layar. Hal ini memungkinkan interaksi langsung dan efisien tanpa perlu perangkat tambahan.

Upload Gambar: Anda juga dapat mengunggah gambar tangan yang membentuk huruf tertentu, dan sistem kami akan menganalisis serta memberikan hasil pengenalan huruf berdasarkan gambar tersebut.

Teknologi di Balik Layar
Kami menggunakan metode Convolutional Neural Network (CNN) yang sudah teruji keakuratannya dalam mengenali gambar. Lebih spesifik lagi, kami mengimplementasikan arsitektur VGG16 dan VGG19, yang terkenal dalam komunitas kecerdasan buatan karena keunggulannya dalam klasifikasi gambar. Dengan model ini, website kami dapat mengenali bahasa isyarat dengan tingkat akurasi yang tinggi, bahkan dalam kondisi pencahayaan yang berbeda atau sudut pandang tangan yang bervariasi.

Manfaat dan Keunggulan
Aksesibilitas: Membuka pintu komunikasi bagi mereka yang memiliki kesulitan mendengar dan berbicara, serta bagi siapa saja yang ingin belajar bahasa isyarat.
Kemudahan Penggunaan: Baik melalui kamera real-time maupun unggah gambar, proses pengenalan bahasa isyarat menjadi lebih mudah dan praktis.
Keandalan: Dengan dukungan teknologi CNN dan model VGG16/VGG19, pengenalan huruf isyarat dapat dilakukan dengan presisi tinggi.

Misi Kami
Kami percaya bahwa teknologi dapat menjadi jembatan untuk menciptakan dunia yang lebih inklusif. Dengan platform ini, kami berharap dapat membantu lebih banyak orang untuk belajar dan memahami bahasa isyarat, sehingga dapat berkomunikasi dengan lebih baik dengan komunitas disabilitas
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
    # Bagian Judul
    st.markdown("## Pengenalan Bahasa Isyarat SIBI")
    st.write("""
    Selamat datang di platform kami yang bertujuan untuk memudahkan komunikasi dengan penyandang disabilitas melalui pengenalan bahasa isyarat SIBI (Sistem Isyarat Bahasa Indonesia). Bahasa isyarat adalah bentuk komunikasi yang menggunakan gerakan tangan untuk menyampaikan pesan, terutama digunakan oleh teman-teman kita yang memiliki hambatan pendengaran.
    """)

    # Bagian Cara Kerja
    st.markdown("## Cara Kerja")
    st.write("""
    Website kami dirancang untuk membantu Anda mengenali huruf-huruf dalam bahasa isyarat SIBI dengan mudah dan cepat. Terdapat dua metode yang bisa Anda gunakan:
    """)
    st.markdown("- **Penggunaan Kamera Real-Time:** Cukup aktifkan kamera Anda, dan sistem kami akan langsung mengenali gerakan tangan Anda. Dalam waktu singkat, huruf yang Anda isyaratkan akan muncul di layar. Hal ini memungkinkan interaksi langsung dan efisien tanpa perlu perangkat tambahan.")
    st.markdown("- **Upload Gambar:** Anda juga dapat mengunggah gambar tangan yang membentuk huruf tertentu, dan sistem kami akan menganalisis serta memberikan hasil pengenalan huruf berdasarkan gambar tersebut.")

    # Bagian Teknologi di Balik Layar
    st.markdown("## Teknologi di Balik Layar")
    st.write("""
    Kami menggunakan metode Convolutional Neural Network (CNN) yang sudah teruji keakuratannya dalam mengenali gambar. Lebih spesifik lagi, kami mengimplementasikan arsitektur VGG16 dan VGG19, yang terkenal dalam komunitas kecerdasan buatan karena keunggulannya dalam klasifikasi gambar. Dengan model ini, website kami dapat mengenali bahasa isyarat dengan tingkat akurasi yang tinggi, bahkan dalam kondisi pencahayaan yang berbeda atau sudut pandang tangan yang bervariasi.
    """)

    # Bagian Manfaat dan Keunggulan
    st.markdown("## Manfaat dan Keunggulan")
    st.write("""
    - **Aksesibilitas:** Membuka pintu komunikasi bagi mereka yang memiliki kesulitan mendengar dan berbicara, serta bagi siapa saja yang ingin belajar bahasa isyarat.
    - **Kemudahan Penggunaan:** Baik melalui kamera real-time maupun unggah gambar, proses pengenalan bahasa isyarat menjadi lebih mudah dan praktis.
    - **Keandalan:** Dengan dukungan teknologi CNN dan model VGG16/VGG19, pengenalan huruf isyarat dapat dilakukan dengan presisi tinggi.
    """)

    # Bagian Misi Kami
    st.markdown("## Misi Kami")
    st.write("""
    Kami percaya bahwa teknologi dapat menjadi jembatan untuk menciptakan dunia yang lebih inklusif. Dengan platform ini, kami berharap dapat membantu lebih banyak orang untuk belajar dan memahami bahasa isyarat, sehingga dapat berkomunikasi dengan lebih baik dengan komunitas disabilitas.
    """)
# Fungsi untuk contoh data gambar
def contoh_gestur():
    st.title("Contoh Gestur Tangan")
    st.write("Berikut adalah contoh gestur tangan untuk setiap huruf dalam alfabet Bahasa Isyarat Indonesia (SIBI):")

    cols = st.columns(4)
    for idx, (label, path) in enumerate(image_paths.items()):
        col = cols[idx % 4]
        with col:
            st.write(f"Mewakili huruf: {label}")
            if os.path.exists(path):
                img = Image.open(path)
                img = correct_image_orientation(img)
                st.image(img, use_column_width=True)
            else:
                st.write(f"File '{path}' tidak ditemukan.")

# Fungsi untuk memuat model menggunakan cache
@st.cache_resource
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        # st.write("Model berhasil dimuat.")
        return model
    except Exception as e:
        st.write(f"Error memuat model: {e}")
        return None

# Jalur file lokal untuk model
vgg16_path = 'model/VGG16FineTune.keras'
vgg19_path = 'model/VGG19FineTune.keras'

# Memuat model dengan cache
model1 = load_model(vgg16_path)
model2 = load_model(vgg19_path)
models = {"VGG16": model1, "VGG19": model2}

def webcam_classification_page(models):
    st.title("Webcam Classification")
    st.write("Use your webcam to classify images using a chosen model.")

    model_choice = st.selectbox("Choose a model", models.keys())
    model = models[model_choice]

    if model is None:
        st.error("Failed to load model. Please check the model path and try again.")
        return
    run = st.button("Start Webcam")
    FRAME_WINDOW = st.image([])

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0)
    detected_letters = []
    last_detected_time = time.time()  # Waktu terakhir huruf terdeteksi

    kalimat_placeholder = st.empty()  # Tempat untuk kalimat yang dibentuk
    predicted_label = ''  # Nilai default untuk predicted_label
    if not camera.isOpened():
        st.error("Failed to access webcam. Please ensure the webcam is properly connected.")
        return

    # Memulai kamera
    while run:
        ret, frame = camera.read()
        if not ret:
            st.write("Failed to capture image from webcam.")
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

                    try:
                        prediction = model.predict(hand_img)
                        predicted_index = np.argmax(prediction)
                        predicted_label = index_to_label[predicted_index]
                    except Exception as e:
                        st.write(f"Error during prediction: {e}")
                        predicted_label = "Error"

                    cv2.putText(frame_rgb, predicted_label, (cx_min, cy_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    current_time = time.time()
                    if current_time - last_detected_time > 2:  # Jeda 2 detik antar huruf
                        detected_letters.append(predicted_label)
                        last_detected_time = current_time

        FRAME_WINDOW.image(frame_rgb)
        kalimat_placeholder.write(f"Kalimat yang dibentuk: {''.join(detected_letters)}  |  Huruf yang terdeteksi: {predicted_label}")  # Menampilkan kalimat dan huruf di sebelah kanan

    camera.release()


def upload_classification_page(models):
    st.title("Image Upload Classification")
    st.write("Upload an image to classify it using both models.")

    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = correct_image_orientation(image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        img_array = np.array(image.resize((128, 128)))/255.0
        img_array = np.expand_dims(img_array, axis=0)
        if models["VGG16"] is None or models["VGG19"] is None:
            st.error("Failed to load one or both models. Please check the model paths and try again.")
            return        
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
    page = st.sidebar.radio("Go to", ["Tentang SIBI", "Contoh Gestur Tangan", "Webcam Classification", "Image Upload Classification"])

    if page == "Tentang SIBI":
        landing_page()
    elif page == "Contoh Gestur Tangan":
        contoh_gestur()
    elif page == "Webcam Classification":
        webcam_classification_page(models)
    elif page == "Image Upload Classification":
        upload_classification_page(models)

if __name__ == "__main__":
    main()